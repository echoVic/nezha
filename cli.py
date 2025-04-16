"""
Typer CLI 定义
"""
import typer

app = typer.Typer()

@app.command()
def main(prompt: str = typer.Argument(..., help="输入你的指令")):
    """nezha 主命令入口"""
    # TODO: 调用 Agent 执行主流程
    typer.echo(f"收到指令: {prompt}")

@app.command()
def plan(initial_requirement: str = typer.Argument(..., help="初始需求描述")):
    """nezha 规划命令入口"""
    # TODO: 交互式规划流程
    typer.echo(f"开始规划: {initial_requirement}")