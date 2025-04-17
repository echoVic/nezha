from typing import List, Optional


class ChatCommand:
    def __init__(self, agent, verbose: bool = False):
        self.agent = agent
        self.verbose = verbose
        self.history = []  # [{role: "user"/"assistant", content: str}]

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if self.verbose:
            prefix = "[用户]" if role == "user" else "[AI]"
            print(f"{prefix}: {content}")

    def interactive_loop(self, initial_message: str = None):
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
                response = self.agent.plan_chat(self.history, self.verbose)
                self.add_message("assistant", response)
                print(f"\n[AI]: {response}")
            except Exception as e:
                print(f"\n[错误] 对话处理出错: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
            
            # 获取下一轮用户输入
            user_input = input("\n请输入你的问题 (输入'exit'退出): ").strip()
            if user_input.lower() in ['exit', 'quit', '退出']:
                break
            self.add_message("user", user_input)

    def run(self, initial_message: str = None):
        """运行对话命令"""
        print("\n开始与AI助手对话，输入'exit'可随时退出\n")
        try:
            self.interactive_loop(initial_message)
            print("\n对话已结束")
        except KeyboardInterrupt:
            print("\n对话被用户中断")
        except Exception as e:
            print(f"\n对话过程中发生错误: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
        return "对话已完成"