"""
Typer CLI 定义
"""
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional

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

from agent import NezhaAgent
from context_engine import ContextEngine
from plan_command import PlanCommand
from security import SecurityLevel, SecurityManager

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
    from cli import models as models_func
    models_func(set_model=True)


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
    table.add_column("API Key", style="dim")
    table.add_column("来源", style="magenta")

    for i, model in enumerate(all_models, start=1):
        name = model.get('name') or model.get('id')
        if model['id'] == current_model_id:
            name = f"[bold]{name} [当前][/bold]"
        source = "预置" if model in PREDEFINED_MODELS else "用户配置"
        table.add_row(
            str(i),
            name,
            model.get('provider', ''),
            model.get('id', ''),
            (model.get('api_key', '')[:6] + '...' if model.get('api_key') else ''),
            source
        )
    console.print(Panel(table, title="nezha models", border_style="blue"))

    if set_model:
        if not all_models:
            console.print("[red]未找到可用模型，请先在 config.yaml 中添加 models 列表！[/red]")
            return
        choice = Prompt.ask(
            "\n请选择模型编号", 
            choices=[str(i) for i in range(1, len(all_models) + 1)],
            default="1"
        )
        selected_index = int(choice) - 1
        selected_model = all_models[selected_index]
        # 只更新 llm.model 字段
        if "llm" not in config:
            config["llm"] = {}
        config["llm"]["model"] = selected_model["id"]
        # 只有当选择的是用户模型时才写入 models
        if selected_model not in PREDEFINED_MODELS:
            # 如果 models 列表没有该用户模型则追加
            if not any(m["id"] == selected_model["id"] for m in user_models):
                user_models.append(selected_model)
                config["models"] = user_models
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)
        console.print(f"\n[green]成功设置模型: {selected_model.get('name', selected_model['id'])}[/green]")
        console.print(f"[dim]配置文件已更新: {config_path}[/dim]")
    else:
        console.print("\n使用 'nezha models --set' 命令可以选择并设置默认模型")

def models_add():
    """添加新模型到 config.yaml 的 models 列表"""
    config_path = get_user_config_path()
    config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    user_models = config.get("models", [])

    console.print("\n[bold cyan]添加新的模型配置[/bold cyan]")
    model_id = Prompt.ask("模型 ID (唯一标识)")
    name = Prompt.ask("模型名称", default=f"用户模型 - {model_id}")
    provider = Prompt.ask("模型提供商", choices=["volcengine", "openai", "anthropic", "wenxin", "custom"], default="volcengine")
    api_key = Prompt.ask("API Key")
    endpoint = Prompt.ask("API Endpoint")

    new_model = {
        "id": model_id,
        "name": name,
        "provider": provider,
        "api_key": api_key,
        "endpoint": endpoint
    }
    # 检查是否已存在
    if any(m["id"] == model_id for m in user_models):
        console.print(f"[red]模型 ID '{model_id}' 已存在，不可重复添加！[/red]")
        return
    user_models.append(new_model)
    config["models"] = user_models
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    console.print(f"[green]模型 '{name}' 已添加到配置文件！[/green]")

@app.command()
def main(
    prompt: str = typer.Argument(..., help="输入你的自然语言指令"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="显示详细执行信息"),
    yes: bool = typer.Option(False, "--yes", "-y", help="自动确认所有操作（谨慎使用）"),
    files: Optional[List[str]] = typer.Option(None, "--file", "-f", help="指定要包含在上下文中的文件"),
    security_level: int = typer.Option(2, "--security", "-s", help="安全级别设置"),
    config_file: Optional[Path] = typer.Option(None, "--config", help="配置文件路径"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="通过命令行注入大模型 API Key，优先级高于配置文件和环境变量")
):
    """nezha 主命令入口 - 执行用户给出的任务指令"""
    # 默认读取用户级别配置
    if config_file is None:
        config_file = get_user_config_path()
        
    # 显示任务信息
    console.print(Panel(f"[bold]执行指令:[/bold] {prompt}", title="nezha", border_style="blue"))
    
    # 初始化安全管理器
    security_manager = SecurityManager(SecurityLevel(security_level), yes_to_all=yes)
    
    # 初始化上下文引擎
    context_engine = ContextEngine(working_dir=os.getcwd())
    
    # 初始化Agent
    agent = NezhaAgent(security_manager=security_manager, config_file=config_file, api_key=api_key)
    
    # 显示进度
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        # 收集上下文
        progress.add_task("收集上下文信息...", total=None)
        context = context_engine.collect(user_files=files)
        
        # 执行指令
        progress.add_task("思考并执行指令...", total=None)
        result = agent.run(prompt, context=context, verbose=verbose)
    
    # 输出结果
    if isinstance(result, str):
        if result.startswith("```markdown") and result.endswith("```"):
            # 如果结果是Markdown格式，使用rich渲染
            md_content = result.replace("```markdown", "").replace("```", "").strip()
            console.print(Markdown(md_content))
        elif result.startswith("```") and result.endswith("```"):
            # 处理代码块
            code_parts = result.split("```", 2)
            if len(code_parts) >= 3:
                lang = code_parts[1].strip()
                code = code_parts[2].strip()
                syntax = Syntax(code, lang, theme="monokai", line_numbers=True)
                console.print(Panel(syntax, title=f"生成的{lang}代码", border_style="green"))
            else:
                console.print(result)
        else:
            # 普通输出
            console.print(Panel(result, title="执行结果", border_style="green"))
    elif isinstance(result, dict) and "table_data" in result:
        # 处理表格数据
        table = Table(title=result.get("title", "结果表格"))
        for column in result["columns"]:
            table.add_column(column)
        for row in result["table_data"]:
            table.add_row(*row)
        console.print(table)
    else:
        # 其他类型输出
        console.print(result)

@app.command()
def plan(
    initial_requirement: str = typer.Argument(..., help="初始需求描述"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="显示详细执行信息"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="输出计划文档的文件路径"),
    config_file: Optional[Path] = typer.Option(None, "--config", help="配置文件路径"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="通过命令行注入大模型 API Key，优先级高于配置文件和环境变量")
):
    """nezha 规划命令入口 - 通过交互式对话生成任务计划"""
    # 显示开始规划的信息
    console.print(Panel(f"[bold]开始规划:[/bold] {initial_requirement}", title="nezha plan", border_style="blue"))
    console.print("[italic]请通过交互式对话完善你的需求...[/italic]")
    
    try:
        # 初始化组件
        security_level = SecurityLevel.NORMAL
        if config_file and config_file.exists():
            # TODO: 从配置文件加载安全级别
            pass
            
        security_manager = SecurityManager(security_level) 
        context_engine = ContextEngine(working_dir=os.getcwd()) 
        
        # 初始化Agent
        agent = NezhaAgent(security_manager=security_manager, config_file=config_file, api_key=api_key)

        # 显示进度
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]收集项目上下文..."),
            console=console,
            transient=True
        ) as progress:
            progress.add_task("收集中", total=None)
            context = context_engine.collect()
            
        # 初始化规划命令
        planner = PlanCommand(
            agent=agent,
            context_engine=context_engine,
            verbose=verbose,
            output_file=output_file
        )

        # 执行交互式规划
        final_plan = planner.run(initial_requirement)
        
        # 显示完成信息
        if output_file:
            console.print(f"\n[bold green]✓[/bold green] 规划已完成，计划文档已保存至: [bold]{output_file}[/bold]")
        else:
            console.print(f"\n[bold green]✓[/bold green] 规划已完成，计划文档已保存至: [bold]plan_output.md[/bold]")
            
        # 显示计划内容预览
        console.print("\n[bold]计划预览:[/bold]")
        console.print(Panel(Markdown(final_plan), title="最终计划", border_style="green"))

    except Exception as e:
        console.print(Panel(f"[bold]执行规划时出错:[/bold] {e}", title="错误", border_style="red"))
        raise typer.Exit(code=1)


def version_callback(value: bool):
    if value:
        console.print("[bold cyan]nezha[/bold cyan] 版本 0.1.0")
        raise typer.Exit()

def get_user_config_path() -> Path:
    """获取用户级别的 nezha 配置文件路径 (~/.config/nezha/config.yaml)"""
    if user_config_dir is not None:
        config_dir = Path(user_config_dir("nezha"))
    else:
        config_dir = Path.home() / ".config" / "nezha"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.yaml"

def get_user_security_config_path() -> Path:
    """获取用户级别的 nezha 安全配置文件路径 (~/.config/nezha/security_config.yaml)"""
    if user_config_dir is not None:
        config_dir = Path(user_config_dir("nezha"))
    else:
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
    # 兼容老参数，优先命令行参数，否则用标准目录
    config_file = config_file or get_user_config_path()
    security_config = security_config or get_user_security_config_path()

    # 显示初始化信息
    console.print(Panel("[bold]初始化nezha配置[/bold]", title="nezha init", border_style="blue"))
    
    # 初始化LLM配置
    llm_config = {}
    console.print("\n[bold]配置大模型接口[/bold]")
    
    # 只允许火山引擎，其他模型后续支持
    console.print("\n[bold yellow]当前仅支持火山引擎 (VolcEngine)，其他模型将在后续版本开放。[/bold yellow]")
    provider = "volcengine"
    llm_config["provider"] = provider
    # TODO: 未来支持其他模型时，恢复多模型选择逻辑
    
    # 配置API密钥
    # 对火山引擎特殊处理提示，因为密钥通常来自环境变量
    api_key_prompt = f"输入 {provider} 的 API 密钥"
    if provider == "volcengine":
        api_key_prompt += " (可选, 优先从环境变量 ARK_API_KEY 读取)"
    api_key = typer.prompt(api_key_prompt, hide_input=True, default="" if provider == "volcengine" else ..., show_default=False)
    # 只有在用户明确输入时才保存到配置，特别是对于火山引擎
    if api_key:
        llm_config["api_key"] = api_key
    
    # 配置模型
    default_models = {
        "openai": "gpt-4o",
        "azure": "gpt-4",
        "anthropic": "claude-3-opus",
        "wenxin": "ernie-bot-4", # 示例模型
        "tongyi": "qwen-max",    # 示例模型
        "zhipuai": "glm-4",     # 示例模型
        "volcengine": "doubao-1-5-pro-32k-250115", # 火山引擎示例模型
        "other": ""
    }
    model = typer.prompt("输入模型名称 (对于火山引擎，这是推理接入点 ID)", default=default_models.get(provider, ""))
    llm_config["model"] = model
    
    # 配置API端点
    default_endpoints = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "azure": "https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions",
        "anthropic": "https://api.anthropic.com/v1/messages",
        "wenxin": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions", # 示例端点
        "tongyi": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation", # 示例端点
        "zhipuai": "https://open.bigmodel.cn/api/paas/v4/chat/completions", # 示例端点
        "volcengine": "https://ark.cn-beijing.volces.com/api/v3", # 火山引擎默认端点
        "other": ""
    }
    # 对火山引擎特殊处理，提示用户可以留空使用默认值
    endpoint_prompt = "输入 API 端点"
    if provider == "volcengine":
        endpoint_prompt += f" (留空则使用默认值: {default_endpoints['volcengine']})"
    else:
        endpoint_prompt += " (如果需要)"
    endpoint = typer.prompt(endpoint_prompt, default=default_endpoints.get(provider, ""), show_default=False)
    if endpoint: # 只有在用户输入了端点时才添加到配置中
        llm_config["endpoint"] = endpoint
    
    # 配置温度和最大token
    temperature = typer.prompt("设置temperature参数", type=float, default=0.2)
    max_tokens = typer.prompt("设置最大输出token数", type=int, default=2048)
    llm_config["temperature"] = temperature
    llm_config["max_tokens"] = max_tokens
    
    # 配置安全设置
    console.print("\n[bold]配置安全设置[/bold]")
    security_levels = ["strict", "normal", "relaxed", "bypass"]
    security_level_idx = typer.prompt(
        "选择安全级别", 
        type=int, 
        default=2, 
        show_choices=False,
        show_default=False,
        prompt_suffix="\n1. 严格 (strict)\n2. 标准 (normal)\n3. 宽松 (relaxed)\n4. 跳过确认 (bypass)\n请选择 [1-4]: "
    )
    
    security_level = security_levels[security_level_idx - 1] if 0 < security_level_idx <= len(security_levels) else security_levels[1]
    
    # 仅生成 rules 占位，用户可在 config.yaml 自由填写规则内容
    rules_config = None  # 用于后续写入空占位及注释
    
    # 生成配置文件
    import yaml

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
    if use_rules:
        full_config["rules"] = rules_config

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
    except Exception as e:
        console.print(Panel(f"[bold]保存配置时出错:[/bold] {e}", title="错误", border_style="red"))
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
        # 初始化组件
        security_level = SecurityLevel.NORMAL
        if config_file and config_file.exists():
            # TODO: 从配置文件加载安全级别
            pass
            
        security_manager = SecurityManager(security_level) 
        
        # 初始化Agent
        agent = NezhaAgent(security_manager=security_manager, config_file=config_file, api_key=api_key)

        # 导入ChatCommand
        from chat_command import ChatCommand

        # 初始化对话命令
        chat_cmd = ChatCommand(
            agent=agent,
            verbose=verbose
        )

        # 执行交互式对话
        chat_cmd.run(initial_message)
        
    except Exception as e:
        console.print(Panel(f"[bold]执行对话时出错:[/bold] {e}", title="错误", border_style="red"))
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