"""
工具注册与调用
"""
import json
from typing import Dict, List, Any, Optional

# 统一导入所有工具类，兼容包内与单文件调试
try:
    from .tools import (
        FileRead, FileWrite, FileEdit,
        Ls,
        Glob, Grep,
        Bash,
        GitStatus, GitLog, GitDiff, GitBranch, GitPull, GitPush
    )
    # 导入新增工具
    from ...features.tools.weather_time import GetCurrentWeather, GetCurrentTime
except ImportError:
    # 兼容直接运行本文件的情况
    from ...features.tools import (
        FileRead, FileWrite, FileEdit,
        Ls,
        Glob, Grep,
        Bash,
        GitStatus, GitLog, GitDiff, GitBranch, GitPull, GitPush
    )
    # 导入新增工具
    from ...features.tools.weather_time import GetCurrentWeather, GetCurrentTime

class ToolRegistry:
    def __init__(self):
        self.tools = {}
        # 注册基础工具
        for tool in [FileRead(), FileWrite(), FileEdit(), Ls(), Glob(), Grep(), Bash(),
                GitStatus(), GitLog(), GitDiff(), GitBranch(), GitPull(), GitPush(),
                GetCurrentWeather(), GetCurrentTime()]:  # 添加新工具
            self.register(tool)

    def register(self, tool):
        self.tools[tool.name] = tool

    def get(self, name):
        return self.tools.get(name)
    
    def get_all_tools(self):
        """获取所有已注册的工具"""
        return list(self.tools.values())
    
    def get_function_schemas(self):
        """获取所有工具的 Function Call 格式定义"""
        from ...features.function_call.function_adapter import tool_to_function_schema
        return [tool_to_function_schema(tool) for tool in self.tools.values()]

def run_tool(tool_name, args):
    registry = ToolRegistry()
    tool = registry.get(tool_name)
    if not tool:
        return f"工具 {tool_name} 未注册"
    try:
        return tool.execute(**args)
    except Exception as e:
        return f"工具执行失败: {e}"