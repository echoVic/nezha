"""
Typer CLI 定义
"""
import os
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

from agent import NezhaAgent
from context_engine import ContextEngine
from plan_command import PlanCommand
from security import SecurityLevel, SecurityManager

app = typer.Typer(help="nezha - AI 命令行代码助手")
console = Console()

@app.command()
def main(
    prompt: str = typer.Argument(..., help="输入你的自然语言指令"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="显示详细执行信息"),
    yes: bool = typer.Option(False, "--yes", "-y", help="自动确认所有操作（谨慎使用）"),
    files: Optional[List[str]] = typer.Option(None, "--file", "-f", help="指定要包含在上下文中的文件"),
    security_level: SecurityLevel = typer.Option(SecurityLevel.NORMAL, "--security", "-s", help="安全级别设置"),
    config_file: Optional[Path] = typer.Option(None, "--config", help="配置文件路径")
):
    """nezha 主命令入口 - 执行用户给出的任务指令"""
    # 显示任务信息
    console.print(Panel(f"[bold]执行指令:[/bold] {prompt}", title="nezha", border_style="blue"))
    
    # 初始化安全管理器
    security_manager = SecurityManager(security_level, auto_confirm=yes)
    
    # 初始化上下文引擎
    context_engine = ContextEngine(working_dir=os.getcwd())
    
    # 初始化Agent
    agent = NezhaAgent(security_manager=security_manager, config_file=config_file)
    
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
    config_file: Optional[Path] = typer.Option(None, "--config", help="配置文件路径")
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
        agent = NezhaAgent(security_manager=security_manager, config_file=config_file)

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

@app.command()
def init(
    config_file: Optional[Path] = typer.Option(
        Path("config/config.yaml"), 
        "--config", 
        "-c", 
        help="配置文件路径"
    ),
    security_config: Optional[Path] = typer.Option(
        Path("config/security_config.yaml"), 
        "--security-config", 
        "-s", 
        help="安全配置文件路径"
    )
):
    """nezha 初始化命令 - 配置大模型接口、token和规则集"""
    # 显示初始化信息
    console.print(Panel("[bold]初始化nezha配置[/bold]", title="nezha init", border_style="blue"))
    
    # 确保配置目录存在
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # 初始化LLM配置
    llm_config = {}
    console.print("\n[bold]配置大模型接口[/bold]")
    
    # 选择LLM提供商
    providers = ["openai", "azure", "anthropic", "other"]
    provider_idx = typer.prompt(
        "选择大模型提供商", 
        type=int, 
        default=1, 
        show_choices=False,
        show_default=False,
        prompt_suffix="\n1. OpenAI\n2. Azure OpenAI\n3. Anthropic\n4. 其他\n请选择 [1-4]: "
    )
    
    provider = providers[provider_idx - 1] if 0 < provider_idx <= len(providers) else providers[0]
    llm_config["provider"] = provider
    
    # 配置API密钥
    api_key = typer.prompt(f"输入{provider}的API密钥", hide_input=True)
    llm_config["api_key"] = api_key
    
    # 配置模型
    default_models = {
        "openai": "gpt-4o",
        "azure": "gpt-4",
        "anthropic": "claude-3-opus",
        "other": ""
    }
    model = typer.prompt("输入模型名称", default=default_models.get(provider, ""))
    llm_config["model"] = model
    
    # 配置API端点
    default_endpoints = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "azure": "https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions",
        "anthropic": "https://api.anthropic.com/v1/messages",
        "other": ""
    }
    endpoint = typer.prompt("输入API端点", default=default_endpoints.get(provider, ""))
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
    
    # 配置规则集
    console.print("\n[bold]配置规则集[/bold]")
    use_rules = typer.confirm("是否配置特定规则集?", default=False)
    rules_config = {}
    
    if use_rules:
        rule_types = ["windsurfrules", "cursorrules", "custom"]
        rule_type_idx = typer.prompt(
            "选择规则集类型", 
            type=int, 
            default=1, 
            show_choices=False,
            show_default=False,
            prompt_suffix="\n1. windsurfrules\n2. cursorrules\n3. 自定义规则\n请选择 [1-3]: "
        )
        
        rule_type = rule_types[rule_type_idx - 1] if 0 < rule_type_idx <= len(rule_types) else rule_types[0]
        rules_config["type"] = rule_type
        
        if rule_type == "custom":
            rules_path = typer.prompt("输入自定义规则文件路径")
            rules_config["path"] = rules_path
    
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
        with open(config_file, "w") as f:
            yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)
        
        with open(security_config, "w") as f:
            yaml.dump(security_config_data, f, default_flow_style=False, sort_keys=False)
        
        console.print(f"\n[bold green]✓[/bold green] 配置已保存至: [bold]{config_file}[/bold] 和 [bold]{security_config}[/bold]")
        console.print("\n现在你可以使用 [bold]nezha <指令>[/bold] 来执行任务了!")
    except Exception as e:
        console.print(Panel(f"[bold]保存配置时出错:[/bold] {e}", title="错误", border_style="red"))
        raise typer.Exit(code=1)


@app.callback(invoke_without_command=True)
def callback(version: bool = typer.Option(False, "--version", "-V", help="显示版本信息", callback=version_callback)):
    """nezha - 基于AI的命令行代码助手"""
    # 只在没有子命令时显示欢迎信息
    ctx = typer.get_app_ctx()
    if ctx.invoked_subcommand is None and not version:
        console.print(
            "[bold cyan]nezha[/bold cyan] - [italic]AI命令行代码助手[/italic] 🚀\n",
            "使用 [bold]nezha <指令>[/bold] 执行任务，[bold]nezha plan <需求>[/bold] 进行交互式规划，或 [bold]nezha init[/bold] 初始化配置\n"
        )
        console.print("运行 [bold]nezha --help[/bold] 获取更多帮助信息")

if __name__ == "__main__":
    app()