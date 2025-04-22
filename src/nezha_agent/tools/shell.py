"""
Bash 命令相关工具
"""
import subprocess

from .base import BaseTool


class Bash(BaseTool):
    name = "Bash"
    description = "在 macOS 终端执行 Shell 命令（高风险，需严格确认）"
    arguments = {"command": "Shell 命令字符串"}
    def execute(self, command):
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            return result.stdout or result.stderr
        except Exception as e:
            return f"执行出错: {e}"
