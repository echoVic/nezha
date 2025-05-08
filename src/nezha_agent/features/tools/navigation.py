"""
目录导航工具
"""
import os

from .base import BaseTool


class Ls(BaseTool):
    name = "Ls"
    description = "列出目录内容"
    arguments = {"path": "目录路径"}
    def execute(self, path="."):
        try:
            return os.listdir(path)
        except Exception as e:
            return f"列出目录失败: {e}"
