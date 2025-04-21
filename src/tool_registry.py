"""
工具注册与调用
"""

from src.tools.file_io import FileRead, FileWrite, FileEdit
from src.tools.navigation import Ls
from src.tools.search import Glob, Grep
from src.tools.shell import Bash
from src.tools.git_tools import GitStatus, GitLog, GitDiff, GitBranch, GitPull, GitPush

class ToolRegistry:
    def __init__(self):
        self.tools = {}
        # 注册基础工具
        for tool in [FileRead(), FileWrite(), FileEdit(), Ls(), Glob(), Grep(), Bash(),
                GitStatus(), GitLog(), GitDiff(), GitBranch(), GitPull(), GitPush()]:
            self.register(tool)

    def register(self, tool):
        self.tools[tool.name] = tool

    def get(self, name):
        return self.tools.get(name)

def run_tool(tool_name, args):
    registry = ToolRegistry()
    tool = registry.get(tool_name)
    if not tool:
        return f"工具 {tool_name} 未注册"
    try:
        return tool.execute(**args)
    except Exception as e:
        return f"工具执行失败: {e}"