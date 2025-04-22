import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class PlanCommand:
    def __init__(self, agent, context_engine, verbose: bool = False, output_file: Optional[Path] = None):
        self.agent = agent
        self.context_engine = context_engine
        self.verbose = verbose
        self.output_file = output_file or Path("plan_output.md")
        self.history: List[Dict[str, str]] = []  # [{role: "user"/"assistant", content: str}]

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if self.verbose:
            prefix = "[用户]" if role == "user" else "[AI]"
            print(f"{prefix}: {content}")

    def interactive_loop(self, initial_requirement: str):
        self.add_message("user", initial_requirement)
        while True:
            # 与agent多轮对话
            response = self.agent.plan_chat(self.history)
            self.add_message("assistant", response)
            print("\nAI建议:\n", response)
            user_input = input("请输入补充/修改需求，或直接回车结束: ").strip()
            if not user_input:
                break
            self.add_message("user", user_input)
        return self.generate_plan_markdown()

    def generate_plan_markdown(self) -> str:
        # 结构化输出Markdown计划
        plan_sections = [msg["content"] for msg in self.history if msg["role"] == "assistant"]
        markdown = "# 项目规划计划\n\n"
        for idx, section in enumerate(plan_sections, 1):
            markdown += f"## 步骤{idx}\n{section}\n\n"
        if self.verbose:
            # print("\n最终计划Markdown:\n", markdown)
            pass
        return markdown

    def save_plan(self, markdown: str):
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(markdown)
        if self.verbose:
            # print(f"计划已保存到: {self.output_file}")
            pass

    def run(self, initial_requirement: str):        
        markdown = self.interactive_loop(initial_requirement)
        self.save_plan(markdown)
        # print(f"\n规划已完成，计划文档输出至: {self.output_file}")
        return markdown