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

@app.callback(invoke_without_command=True)
def callback(version: bool = typer.Option(False, "--version", "-V", help="显示版本信息", callback=version_callback)):
    """nezha - 基于AI的命令行代码助手"""
    # 只在没有子命令时显示欢迎信息
    ctx = typer.get_app_ctx()
    if ctx.invoked_subcommand is None and not version:
        console.print(
            "[bold cyan]nezha[/bold cyan] - [italic]AI命令行代码助手[/italic] 🚀\n",
            "使用 [bold]nezha <指令>[/bold] 执行任务，或 [bold]nezha plan <需求>[/bold] 进行交互式规划\n"
        )
        console.print("运行 [bold]nezha --help[/bold] 获取更多帮助信息")

if __name__ == "__main__":
    app()