"""
文件搜索与内容查找工具
"""
import glob
import re

from .base import BaseTool


class Glob(BaseTool):
    name = "Glob"
    description = "在指定目录查找匹配模式的文件/目录"
    arguments = {"pattern": "glob 匹配模式", "cwd": "搜索目录"}
    def execute(self, pattern, cwd="."):
        return glob.glob(pattern, root_dir=cwd)

class Grep(BaseTool):
    name = "Grep"
    description = "在文件中搜索内容"
    arguments = {"pattern": "正则表达式", "path": "文件路径"}
    def execute(self, pattern, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            matches = [line for line in lines if re.search(pattern, line)]
            return matches
        except Exception as e:
            return f"搜索出错: {e}"
