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

from ...features.agent.agent import NezhaAgent
from ...core.context.context_engine import ContextEngine
from ...features.commands.plan_command import PlanCommand
from ...core.security.security import SecurityLevel, SecurityManager

app = typer.Typer(
    help="nezha - AI 命令行代码助手\n\n模型管理相关命令：\n  nezha models              查看所有模型并切换当前模型\n  nezha models add          添加新模型到配置文件\n  nezha models list         仅列出所有模型（只读）\n\n其他命令请用 nezha --help 查看。",
    no_args_is_help=True,
    add_completion=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
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
    },
    {
        "id": "qwen3-235b-a22b",
        "model": "qwen3-235b-a22b",
        "name": "阿里千问 - Qwen3-235B",
        "provider": "qwen",
        "api_key": "****",
        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    {
        "id": "qwen-plus",
        "model": "qwen-plus-2025-04-28",
        "name": "阿里千问 - Qwen-Plus",
        "provider": "qwen",
        "api_key": "****",
        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
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
            
            # 完全更新llm部分的配置
            config["llm"] = {
                "model": selected_model["id"],
                "provider": selected_model.get("provider"),
                "api_key": selected_model.get("api_key"),
            }
            
            # 如果有endpoint，也添加到配置中
            if "endpoint" in selected_model:
                config["llm"]["endpoint"] = selected_model["endpoint"]
            
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
    
    model = Prompt.ask("模型名称/ID")
    api_key = Prompt.ask("API Key")
    endpoint = Prompt.ask("API 端点", default="")
    
    # 创建模型配置
    new_model = {
        "id": model,  # 使用模型名称作为内部ID
        "model": model,  # 同时设置 model 字段
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
    stream: bool = typer.Option(True, "--stream", help="启用流式输出（默认开启）"),
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
        
        if not stream:
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
        else:
            # 流式输出模式
            console.print("\n[bold cyan]nezha:[/bold cyan]")
            console.print("[dim](按Ctrl+C可中断生成)[/dim]")
            # 执行任务并获取生成器
            response_generator = agent.run(prompt, verbose=verbose, stream=True)
            # 逐个输出响应片段
            response_text = ""
            try:
                for chunk in response_generator:
                    console.print(chunk, end="")
                    response_text += chunk
            except KeyboardInterrupt:
                console.print("\n\n[bold red][生成已被用户中断][/bold red]")
            finally:
                console.print()  # 换行
                # 包装结果
                result = {"response": response_text, "error": None, "tool_calls": []}
        
        # 显示结果
        if result.get("error"):
            console.print(Panel(f"[bold]执行出错:[/bold] {result['error']}", title="错误", border_style="red"))
        elif not stream:  # 只在非流式输出模式下显示完整响应
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
    stream: bool = typer.Option(True, "--stream", help="启用流式输出（默认开启）"),
    security_level: str = typer.Option(
        "normal", "--security", "-s",
        help="安全级别: strict(严格), normal(标准), relaxed(宽松), bypass(绕过)"
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
        # 将字符串安全级别转换为枚举类型
        security_level_map = {
            "strict": SecurityLevel.STRICT,
            "normal": SecurityLevel.NORMAL,
            "relaxed": SecurityLevel.RELAXED,
            "bypass": SecurityLevel.BYPASS
        }
        if security_level.lower() not in security_level_map:
            raise typer.BadParameter(f"不支持的安全级别: {security_level}，可选值: strict, normal, relaxed, bypass")
        security_enum = security_level_map[security_level.lower()]
        
        # 初始化安全管理器
        security_manager = SecurityManager(security_enum)
        
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
        
        # 初始化 PlanCommand
        plan_command = PlanCommand(
            agent=agent,
            context_engine=context_engine,
            verbose=verbose,
            stream=stream
        )
        
        # 在非流式输出模式下显示加载进度
        if not stream:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("正在加载模型...", total=None)
                # 执行规划
                plan_command.run(initial_requirement)
        else:
            # 流式输出模式下直接执行规划
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

@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", help="显示版本信息", callback=version_callback, is_eager=True)
):
    """nezha - 基于AI的命令行代码助手"""
    if version:
        return

    if ctx.invoked_subcommand is None:  # No explicit subcommand was called (e.g. init, plan, models)
        if ctx.args:  # Check if there are any unparsed arguments
            initial_message = " ".join(ctx.args)
            if initial_message.strip(): # Ensure it's not just whitespace
                # console.print(f"[Debug in callback] Default invocation with ctx.args: {ctx.args}. Invoking 'main' command...")
                
                config_file_val = ctx.params.get('config_file')
                api_key_val = ctx.params.get('api_key')
                verbose_val = ctx.params.get('verbose', False) 
                stream_val = ctx.params.get('stream', True)

                try:
                    # Prepare parameters for the 'main' command
                    params_for_main = {
                        'prompt': initial_message,
                        'verbose': verbose_val,
                        'stream': stream_val,
                        'config_file': config_file_val,
                        'api_key': api_key_val
                    }
                    # Clean params, but ensure necessary ones for 'main' like 'prompt' are present
                    # For boolean flags, ctx.invoke handles defaults well if not provided.
                    
                    # Invoke the 'main' command
                    ctx.invoke(main, **params_for_main)

                except typer.Exit:
                    raise
                except Exception as e:
                    console.print(Panel(f"[bold]在尝试通过回调调用 main 命令时发生错误:[/bold] {e}", title="回调错误", border_style="red"))
                    raise typer.Exit(code=1)
                return  # Handled, exit callback

        # If no subcommand AND no args (or only whitespace args), show welcome message
        console.print(
            "[bold cyan]nezha[/bold cyan] - [italic]AI命令行代码助手[/italic] 🚀\n",
            "使用 [bold]nezha <指令>[/bold] 执行任务，[bold]nezha plan <需求>[/bold] 进行交互式规划，[bold]nezha chat[/bold] 进行对话，或 [bold]nezha init[/bold] 初始化配置\n"
        )
        console.print("运行 [bold]nezha --help[/bold] 获取更多帮助信息")
    # If ctx.invoked_subcommand is NOT None, Typer will proceed to call the subcommand.

if __name__ == "__main__":
    app()
