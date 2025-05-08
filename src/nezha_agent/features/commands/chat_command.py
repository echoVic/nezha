from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# 使用 rich 库自己的控制台，避免循环导入
from rich.console import Console
console = Console()


class ChatCommand:
    def __init__(self, agent, verbose: bool = False):
        self.agent = agent
        self.verbose = verbose
        self.history = []  # [{role: "user"/"assistant", content: str}]

    def add_message(self, role: str, content: str):
        """添加消息到历史记录"""
        self.history.append({"role": role, "content": content})
        # 如果需要在控制台显示消息，可以取消下面的注释
        # if self.verbose:
        #     display_prefix = "[用户]" if role == "user" else "[nezha]"
        #     console.print(f"{display_prefix}: {content}")

    def interactive_loop(self, initial_message=None):
        """处理交互式对话循环"""
        if initial_message:
            self.add_message("user", initial_message)
        
        while True:
            if not self.history:  # 如果没有初始消息，先获取用户输入
                user_input = input("\n请输入你的问题 (输入'exit'退出): ").strip()
                if user_input.lower() in ['exit', 'quit', '退出']:
                    break
                self.add_message("user", user_input)
            
            # 调用agent处理对话
            try:
                # 添加加载指示器
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True,  # 完成时移除进度显示
                    console=console   # 使用相同的控制台
                ) as progress:
                    progress.add_task(description="思考中...", total=None)  # 不确定的任务
                    response = self.agent.plan_chat(self.history, self.verbose)
                
                self.add_message("assistant", response)
                console.print(Panel(Markdown(response), title="nezha", border_style="cyan"))
            except (ValueError, TypeError) as error:
                # 处理可预期的错误类型
                console.print(Panel(f"[bold]对话处理出错:[/bold] {error}", title="错误", border_style="red"))
                if self.verbose:
                    import traceback
                    console.print(traceback.format_exc())
            except Exception as error:
                # 处理未预期的错误
                console.print(Panel(f"[bold]未预期的错误:[/bold] {error}", title="错误", border_style="red"))
                if self.verbose:
                    import traceback
                    console.print(traceback.format_exc())
            
            # 获取下一轮用户输入
            user_input = input("\n请输入你的问题 (输入'exit'退出): ").strip()
            if user_input.lower() in ['exit', 'quit', '退出']:
                break
            self.add_message("user", user_input)

    def run(self, initial_message=None):
        """运行对话命令"""
        try:
            self.interactive_loop(initial_message)
        except KeyboardInterrupt:
            # 处理用户中断
            pass
        except KeyError as error:
            # 处理键错误（如访问不存在的字典键）
            console.print(Panel(f"[bold]对话数据错误:[/bold] {error}", title="错误", border_style="red"))
            if self.verbose:
                import traceback
                console.print(traceback.format_exc())
        except Exception as error:
            # 处理其他未预期的错误
            console.print(Panel(f"[bold]对话过程中发生错误:[/bold] {error}", title="错误", border_style="red"))
            if self.verbose:
                import traceback
                console.print(traceback.format_exc())
        return "对话已完成"