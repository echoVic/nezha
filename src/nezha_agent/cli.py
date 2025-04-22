"""
Typer CLI 定义
"""
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional             

# 工具函数：自动补全 LLM 配置，确保 provider/api_key/endpoint 存在
def merge_llm_config(config: dict) -> dict:
    """
    根据 config['llm']['model']，自动补全 provider/api_key/endpoint 等字段。
    """
    llm_config = dict(config.get("llm", {}))
    models = config.get("models", [])
    model_id = llm_config.get("model")
    if model_id and isinstance(models, list):
        for m in models:
            if m.get("id") == model_id:
                for k in ["provider", "api_key", "endpoint"]:
                    if k in m and not llm_config.get(k):
                        llm_config[k] = m[k]
                break
    return llm_config

try:
    from platformdirs import user_config_dir
except ImportError:
    user_config_dir = None  # 兼容未安装 platformdirs 的情况

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table

from .agent import NezhaAgent
from .context_engine import ContextEngine
from .plan_command import PlanCommand
from .security import SecurityLevel, SecurityManager

app = typer.Typer(
    help="nezha - AI 命令行代码助手\n\n模型管理相关命令：\n  nezha models              查看所有模型并切换当前模型\n  nezha models add          添加新模型到配置文件\n  nezha models list         仅列出所有模型（只读）\n\n其他命令请用 nezha --help 查看。",
    no_args_is_help=True,
    add_completion=True,
)

console = Console()

models_app = typer.Typer(help="模型管理相关命令")
app.add_typer(models_app, name="models", help="模型管理相关命令")

# 全局变量用于存储当前选择的模型
CURRENT_MODEL = None

# 预定义的模型列表
PREDEFINED_MODELS = [
    {
        "id": "ep-20250417174840-6c94l",
        "name": "火山引擎 - Doubao-1.5-pro-32k",
        "provider": "volcengine",
        "api_key": "****",
        "endpoint": "https://ark.cn-beijing.volces.com/api/v3",
    }
]

# 获取当前选择的模型
def get_current_model():
    """获取当前选择的模型配置。如果没有设置，返回 None。"""
    global CURRENT_MODEL
    return CURRENT_MODEL

# 设置当前选择的模型
def set_current_model(model):
    """设置当前选择的模型配置。"""
    global CURRENT_MODEL
    CURRENT_MODEL = model

@models_app.command("list")
def list_models():
    """仅列出所有模型（只读）"""
    models(
        set_model=False
    )

@models_app.command("select")
def select_model():
    """交互式选择并设置默认模型（推荐使用）"""
    from rich.console import Console
    console = Console()
    # 复用 models 逻辑，进入交互选择模式
    models(set_model=True)


@models_app.command("add")
def add_model():
    """添加新模型到 config.yaml 的 models 列表"""
    models_add()

@models_app.command("__default__")
def models(
    set_model: bool = typer.Option(False, "--set", "-s", help="设置默认模型"),
):
    """列出可用的模型并允许用户选择"""
    config_path = get_user_config_path()
    config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    user_models = config.get("models", [])
    current_model_id = config.get("llm", {}).get("model")

    # 合并展示：预置模型 + 用户模型
    all_models = PREDEFINED_MODELS + user_models

    table = Table(title="可用的模型")
    table.add_column("序号", style="cyan")
    table.add_column("模型名称", style="green")
    table.add_column("提供商", style="yellow")
    table.add_column("模型 ID", style="blue")
    table.add_column("状态", style="magenta")

    for i, model in enumerate(all_models):
        model_id = model.get("id")
        status = "✓ 当前" if model_id == current_model_id else ""
        table.add_row(
            str(i+1),
            model.get("name", "未命名"),
            model.get("provider", "未知"),
            model_id or "未知",
            status
        )
    
    console.print(table)
    
    if set_model:
        # 交互式选择模型
        choice = Prompt.ask(
            "请选择要使用的模型 [序号]",
            choices=[str(i+1) for i in range(len(all_models))],
            default="1"
        )
        
        try:
            selected_index = int(choice) - 1
            selected_model = all_models[selected_index]
            
            # 更新配置文件
            if not config.get("llm"):
                config["llm"] = {}
            
            config["llm"]["model"] = selected_model["id"]
            
            # 确保 models 列表中包含选中的模型
            if selected_index < len(PREDEFINED_MODELS):
                # 选择的是预定义模型，确保添加到用户配置
                model_exists = False
                for m in user_models:
                    if m.get("id") == selected_model["id"]:
                        model_exists = True
                        break
                
                if not model_exists:
                    if "models" not in config:
                        config["models"] = []
                    config["models"].append(selected_model)
            
            # 保存配置
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
            console.print(f"[bold green]✓[/bold green] 已将默认模型设置为: [bold]{selected_model.get('name')}[/bold]")
            
            # 设置当前模型（全局变量）
            set_current_model(selected_model)
            
        except (ValueError, IndexError):
            console.print("[bold red]✗[/bold red] 无效的选择")
            return

@models_app.command("add")
def models_add():
    """添加新模型到 config.yaml 的 models 列表"""
    config_path = get_user_config_path()
    config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    
    # 获取用户输入
    console.print("[bold]添加新模型[/bold]")
    name = Prompt.ask("模型名称")
    provider = Prompt.ask("提供商", choices=["openai", "volcengine", "other"], default="openai")
    if provider == "other":
        provider = Prompt.ask("请输入提供商名称")
    
    model_id = Prompt.ask("模型 ID")
    api_key = Prompt.ask("API Key")
    endpoint = Prompt.ask("API 端点", default="")
    
    # 创建模型配置
    new_model = {
        "id": model_id,
        "name": name,
        "provider": provider,
        "api_key": api_key
    }
    
    if endpoint:
        new_model["endpoint"] = endpoint
    
    # 更新配置
    if "models" not in config:
        config["models"] = []
    
    config["models"].append(new_model)
    
    # 保存配置
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    console.print(f"[bold green]✓[/bold green] 已添加新模型: [bold]{name}[/bold]")

@app.command()
def main(
    prompt: str = typer.Argument(..., help="输入你的自然语言指令"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="显示详细执行信息"),
    security_level: str = typer.Option(
        "normal", "--security", "-s",
        help="安全级别: strict(严格), normal(普通), relaxed(宽松), bypass(绕过)"
    ),
    yes_to_all: bool = typer.Option(False, "--yes", "-y", help="自动确认所有操作"),
    config_file: Optional[Path] = typer.Option(None, "--config", help="配置文件路径"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="通过命令行注入大模型 API Key，优先级高于配置文件和环境变量")
):
    """nezha 主命令入口 - 执行用户给出的任务指令"""
    # 默认读取用户级别配置
    if config_file is None:
        config_file = get_user_config_path()
    
    # 显示任务信息
    console.print(Panel(f"[bold]执行任务:[/bold] {prompt}", title="nezha", border_style="blue"))
    
    # 将字符串安全级别转换为枚举，防止类型混用
    security_level_map = {
        "strict": SecurityLevel.STRICT,
        "normal": SecurityLevel.NORMAL,
        "relaxed": SecurityLevel.RELAXED,
        "bypass": SecurityLevel.BYPASS
    }
    if security_level.lower() not in security_level_map:
        raise typer.BadParameter(f"不支持的安全级别: {security_level}，可选值: strict, normal, relaxed, bypass")
    security_enum = security_level_map[security_level.lower()]
    # 初始化安全管理器，后续所有用到安全等级的地方都只能用 security_enum
    security_manager = SecurityManager(security_enum, yes_to_all=yes_to_all)
    
    try:
        # 初始化上下文引擎
        context_engine = ContextEngine(working_dir=os.getcwd())
        
        # 收集上下文信息
        context_engine.collect()
        
        # 初始化 Agent
        agent = NezhaAgent(
            security_manager=security_manager,
            config_file=config_file,
            api_key=api_key
        )
        
        # 设置上下文引擎
        agent.context_engine = context_engine
        
        # 显示加载进度
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("正在加载模型...", total=None)
            # 执行任务
            result = agent.run(prompt)
            # 兼容 agent.run 返回字符串，包装成 dict
            if not isinstance(result, dict):
                result = {"response": result, "error": None, "tool_calls": []}
        
        # 显示结果
        if result.get("error"):
            console.print(Panel(f"[bold]执行出错:[/bold] {result['error']}", title="错误", border_style="red"))
        else:
            # 显示 AI 回复
            console.print("\n[bold cyan]nezha:[/bold cyan]")
            
            # 尝试解析 Markdown
            try:
                markdown_content = result.get("response", "")
                console.print(Markdown(markdown_content))
            except (ValueError, TypeError) as error:
                # 如果解析失败，直接显示原始文本
                console.print(result.get("response", ""))
            
            # 显示执行的工具调用
            if verbose and result.get("tool_calls"):
                console.print("\n[bold yellow]执行的工具调用:[/bold yellow]")
                for i, call in enumerate(result.get("tool_calls", [])):
                    tool_name = call.get("name", "未知工具")
                    tool_args = call.get("arguments", {})
                    console.print(f"[bold]{i+1}.[/bold] [cyan]{tool_name}[/cyan]")
                    
                    # 格式化显示参数
                    args_syntax = Syntax(
                        yaml.dump(tool_args, default_flow_style=False, sort_keys=False, allow_unicode=True),
                        "yaml",
                        theme="monokai",
                        line_numbers=False,
                    )
                    console.print(args_syntax)
                    
                    # 显示结果
                    if "result" in call:
                        result_syntax = Syntax(
                            str(call["result"]),
                            "text",
                            theme="monokai",
                            line_numbers=False,
                        )
                        console.print(Panel(result_syntax, title="结果", border_style="green"))
    
    except (ValueError, TypeError) as error:
        # 处理可预期的错误类型
        console.print(Panel(f"[bold]执行任务时出错:[/bold] {error}", title="错误", border_style="red"))
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(code=1)
    except Exception as error:
        # 处理未预期的错误
        console.print(Panel(f"[bold]执行任务时出错:[/bold] {error}", title="错误", border_style="red"))
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(code=1)

@app.command()
def plan(
    initial_requirement: str = typer.Argument(..., help="初始需求描述"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="显示详细执行信息"),
    security_level: SecurityLevel = typer.Option(
        SecurityLevel.NORMAL, "--security", "-s",
        help="安全级别: strict(严格), normal(普通), relaxed(宽松), bypass(绕过)"
    ),
    config_file: Optional[Path] = typer.Option(None, "--config", help="配置文件路径"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="通过命令行注入大模型 API Key，优先级高于配置文件和环境变量")
):
    """nezha 规划命令入口 - 通过交互式对话生成任务计划"""
    # 默认读取用户级别配置
    if config_file is None:
        config_file = get_user_config_path()
    
    # 显示任务信息
    console.print(Panel(f"[bold]需求描述:[/bold] {initial_requirement}", title="nezha plan", border_style="blue"))
    
    try:
        # 初始化安全管理器
        security_manager = SecurityManager(security_level)
        
        # 初始化上下文引擎
        context_engine = ContextEngine(working_dir=os.getcwd())
        
        # 收集上下文信息
        context_engine.collect()
        
        # 初始化 Agent
        agent = NezhaAgent(
            security_manager=security_manager,
            config_file=config_file,
            api_key=api_key
        )
        
        # 设置上下文引擎
        agent.context_engine = context_engine
        
        # 显示加载进度
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("正在加载模型...", total=None)
            
            # 初始化 PlanCommand
            plan_command = PlanCommand(
                agent=agent,
                context_engine=context_engine,
                verbose=verbose
            )
            
            # 执行规划
            plan_command.run(initial_requirement)
    
    except (ValueError, TypeError) as error:
        # 处理可预期的错误类型
        console.print(Panel(f"[bold]执行规划时出错:[/bold] {error}", title="错误", border_style="red"))
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(code=1)
    except Exception as error:
        # 处理未预期的错误
        console.print(Panel(f"[bold]执行规划时出错:[/bold] {error}", title="错误", border_style="red"))
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(code=1)


def version_callback(value: bool):
    if value:
        console.print("[bold cyan]nezha[/bold cyan] v0.1.0")
        raise typer.Exit()

# 获取用户级别的 nezha 配置文件路径 (~/.config/nezha/config.yaml)
def get_user_config_path():
    """获取用户级别的 nezha 配置文件路径 (~/.config/nezha/config.yaml)"""
    if user_config_dir:
        config_dir = Path(user_config_dir("nezha", "nezha"))
    else:
        # 兼容未安装 platformdirs 的情况
        config_dir = Path.home() / ".config" / "nezha"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.yaml"

# 获取用户级别的 nezha 安全配置文件路径 (~/.config/nezha/security_config.yaml)
def get_user_security_config_path():
    """获取用户级别的 nezha 安全配置文件路径 (~/.config/nezha/security_config.yaml)"""
    if user_config_dir:
        config_dir = Path(user_config_dir("nezha", "nezha"))
    else:
        # 兼容未安装 platformdirs 的情况
        config_dir = Path.home() / ".config" / "nezha"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "security_config.yaml"

@app.command()
def init(
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="配置文件路径，默认写入用户目录 ~/.config/nezha/config.yaml"
    ),
    security_config: Optional[Path] = typer.Option(
        None, "--security-config", "-s", help="安全配置文件路径，默认写入用户目录 ~/.config/nezha/security_config.yaml"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="通过命令行注入大模型 API Key，优先级高于配置文件和环境变量"
    ),
):
    """nezha 初始化命令 - 配置大模型接口、token和规则集"""
    # 默认读取用户级别配置
    if config_file is None:
        config_file = get_user_config_path()
    
    if security_config is None:
        security_config = get_user_security_config_path()
    
    # 确保目录存在
    config_file.parent.mkdir(parents=True, exist_ok=True)
    security_config.parent.mkdir(parents=True, exist_ok=True)
    
    console.print(Panel("[bold]欢迎使用 nezha 初始化向导[/bold]", title="nezha init", border_style="blue"))
    console.print("这将帮助你配置 nezha 的基本设置，包括大模型接口和安全级别。")
    
    # 读取已有配置（如果存在）
    existing_config = {}
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                existing_config = yaml.safe_load(f) or {}
            console.print(f"[bold yellow]注意:[/bold yellow] 检测到已有配置文件 {config_file}，将在其基础上更新。")
        except (IOError, yaml.YAMLError) as error:
            console.print(f"[bold red]✗[/bold red] 读取现有配置文件失败: {error}")
    
    # 选择模型提供商
    providers = ["openai", "volcengine", "other"]
    provider_names = {
        "openai": "OpenAI API",
        "volcengine": "火山引擎",
        "other": "其他提供商"
    }
    
    console.print("\n[bold]第一步: 选择大模型提供商[/bold]")
    for i, p in enumerate(providers):
        console.print(f"  {i+1}. {provider_names[p]}")
    
    provider_idx = Prompt.ask(
        "请选择提供商 [序号]",
        choices=[str(i+1) for i in range(len(providers))],
        default="1"
    )
    provider = providers[int(provider_idx) - 1]
    
    if provider == "other":
        provider = Prompt.ask("请输入提供商名称")
    
    # 配置 API Key
    console.print(f"\n[bold]第二步: 配置 {provider_names.get(provider, provider)} API Key[/bold]")
    
    # 如果命令行提供了 API Key，优先使用
    if api_key:
        console.print(f"[bold green]✓[/bold green] 已通过命令行参数提供 API Key")
    else:
        # 尝试从环境变量获取
        env_var_name = f"{provider.upper()}_API_KEY"
        env_api_key = os.environ.get(env_var_name)
        
        if env_api_key:
            use_env = Prompt.ask(
                f"检测到环境变量 {env_var_name}，是否使用该值?",
                choices=["y", "n"],
                default="y"
            )
            
            if use_env.lower() == "y":
                api_key = env_api_key
                console.print(f"[bold green]✓[/bold green] 已使用环境变量中的 API Key")
            else:
                api_key = Prompt.ask("请输入 API Key", password=True)
        else:
            api_key = Prompt.ask("请输入 API Key", password=True)
    
    # 配置 API 端点
    endpoint = None
    if provider != "openai":
        console.print(f"\n[bold]第三步: 配置 {provider_names.get(provider, provider)} API 端点[/bold]")
        default_endpoints = {
            "volcengine": "https://ark.cn-beijing.volces.com/api/v3",
        }
        default_endpoint = default_endpoints.get(provider, "")
        endpoint = Prompt.ask("请输入 API 端点", default=default_endpoint)
    
    # 选择默认模型
    console.print("\n[bold]第四步: 选择默认模型[/bold]")
    default_models = {
        "openai": [
            {"id": "gpt-4", "name": "GPT-4"},
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
        ],
        "volcengine": [
            {"id": "ep-20250417174840-6c94l", "name": "Doubao-1.5-pro-32k"},
        ]
    }
    
    models = default_models.get(provider, [])
    if models:
        for i, m in enumerate(models):
            console.print(f"  {i+1}. {m['name']} ({m['id']})")
        
        model_idx = Prompt.ask(
            "请选择默认模型 [序号]",
            choices=[str(i+1) for i in range(len(models))],
            default="1"
        )
        model = models[int(model_idx) - 1]["id"]
    else:
        model = Prompt.ask("请输入模型 ID")
    
    # 选择安全级别
    console.print("\n[bold]第五步: 选择安全级别[/bold]")
    security_levels = [
        {"id": "strict", "name": "严格 (只允许读取文件)"},
        {"id": "normal", "name": "普通 (允许读写文件，需确认高风险操作)"},
        {"id": "relaxed", "name": "宽松 (允许执行 shell 命令，需确认高风险操作)"},
        {"id": "bypass", "name": "绕过 (允许所有操作，不需确认)"}
    ]
    
    for i, level in enumerate(security_levels):
        console.print(f"  {i+1}. {level['name']}")
    
    level_idx = Prompt.ask(
        "请选择安全级别 [序号]",
        choices=[str(i+1) for i in range(len(security_levels))],
        default="2"
    )
    security_level = security_levels[int(level_idx) - 1]["id"]
    
    # 构建 LLM 配置
    llm_config = {
        "provider": provider,
        "api_key": api_key,
        "model": model
    }
    
    if endpoint:
        llm_config["endpoint"] = endpoint
    
    # 生成主配置文件
    full_config = {
        "llm": llm_config,
        "security": {
            "allow_bash": security_level in ["relaxed", "bypass"],
            "allow_file_write": security_level != "strict",
            "allow_file_edit": security_level != "strict",
            "confirm_high_risk": security_level != "bypass"
        },
        "tools": {
            "enabled": [
                "FileRead", 
                "FileWrite", 
                "FileEdit", 
                "Glob", 
                "Grep", 
                "Ls"
            ]
        }
    }
    # 暂时注释掉 rules 配置相关代码，避免 use_rules 未定义错误
    # if use_rules:
    #     full_config["rules"] = rules_config

    # 生成安全配置文件
    security_config_data = {
        "security_level": security_level,
        "yes_to_all": False,
        "allowed_paths": [],
        "disabled_tools": []
    }
    # 写入配置文件
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(full_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        with open(security_config, "w", encoding="utf-8") as f:
            yaml.dump(security_config_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        console.print(f"\n[bold green]✓[/bold green] 配置已保存至: [bold]{config_file}[/bold] 和 [bold]{security_config}[/bold]")
        console.print("\n现在你可以使用 [bold]nezha <指令>[/bold] 来执行任务了!")
    except (IOError, yaml.YAMLError) as error:
        console.print(Panel(f"[bold]保存配置时出错:[/bold] {error}", title="错误", border_style="red"))
        raise typer.Exit(code=1)
    except Exception as error:
        console.print(Panel(f"[bold]未预期的错误:[/bold] {error}", title="错误", border_style="red"))
        raise typer.Exit(code=1)


@app.command()
def chat(
    initial_message: Optional[str] = typer.Argument(None, help="初始对话消息"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="显示详细执行信息"),
    config_file: Optional[Path] = typer.Option(None, "--config", help="配置文件路径"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="通过命令行注入大模型 API Key，优先级高于配置文件和环境变量")
):
    """nezha 对话命令 - 与AI助手进行交互式对话"""
    # 默认读取用户级别配置
    if config_file is None:
        config_file = get_user_config_path()
    # 显示开始对话的信息
    console.print(Panel("[bold]开始与AI助手对话[/bold]", title="nezha chat", border_style="blue"))
    
    try:
        # ========== 启用多轮对话流程 ===========
        # 初始化安全管理器和 Agent
        # 注意：chat 命令暂时默认 normal，如需支持自定义传参可扩展参数
        security_level_map = {
            "strict": SecurityLevel.STRICT,
            "normal": SecurityLevel.NORMAL,
            "relaxed": SecurityLevel.RELAXED,
            "bypass": SecurityLevel.BYPASS
        }
        security_enum = security_level_map["normal"]  # 默认 normal
        security_manager = SecurityManager(security_enum)
        agent = NezhaAgent(security_manager=security_manager, config_file=config_file, api_key=api_key)
        
        # 导入并运行 ChatCommand
        from .chat_command import ChatCommand
        chat_cmd = ChatCommand(
            agent=agent,
            verbose=verbose
        )
        chat_cmd.run(initial_message)
        
    except (ValueError, TypeError) as error:
        console.print(Panel(f"[bold]执行对话时出错:[/bold] {error}", title="错误", border_style="red"))
        raise typer.Exit(code=1)
    except Exception as error:
        console.print(Panel(f"[bold]未预期的错误:[/bold] {error}", title="错误", border_style="red"))
        raise typer.Exit(code=1)


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", help="显示版本信息", callback=version_callback)
):
    """nezha - 基于AI的命令行代码助手"""
    # 只在没有子命令时显示欢迎信息
    if ctx.invoked_subcommand is None and not version:
        console.print(
            "[bold cyan]nezha[/bold cyan] - [italic]AI命令行代码助手[/italic] 🚀\n",
            "使用 [bold]nezha <指令>[/bold] 执行任务，[bold]nezha plan <需求>[/bold] 进行交互式规划，[bold]nezha chat[/bold] 进行对话，或 [bold]nezha init[/bold] 初始化配置\n"
        )
        console.print("运行 [bold]nezha --help[/bold] 获取更多帮助信息")

if __name__ == "__main__":
    app()
