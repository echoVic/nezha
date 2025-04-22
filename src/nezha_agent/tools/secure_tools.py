"""
安全工具集成示例 - 展示如何将安全层与工具系统集成
"""

from typing import Any, Dict, Optional

from nezha_agent.security import SecurityLevel, ToolRiskLevel, security_manager
from .base import BaseTool
from .file_io import FileEdit, FileWrite
from .shell import Bash


class SecureFileWrite(FileWrite):
    """安全版文件写入工具，集成了安全检查"""
    
    def execute(self, path: str, content: str) -> str:
        # 检查路径是否在允许范围内
        if not security_manager.is_path_allowed(path):
            return f"安全错误: 路径 '{path}' 不在允许操作的范围内"
        
        # 请求用户确认
        details = {
            "路径": path,
            "内容长度": f"{len(content)} 字符",
            "内容预览": f"{content[:100]}{'...' if len(content) > 100 else ''}"
        }
        
        if not security_manager.confirm_action(
            message=f"即将写入文件 '{path}'",
            risk_level=ToolRiskLevel.MEDIUM,
            details=details
        ):
            return "操作已取消"
        
        # 调用原始方法执行写入
        return super().execute(path, content)


class SecureFileEdit(FileEdit):
    """安全版文件编辑工具，集成了安全检查"""
    
    def execute(self, path: str, instructions: str) -> str:
        # 检查路径是否在允许范围内
        if not security_manager.is_path_allowed(path):
            return f"安全错误: 路径 '{path}' 不在允许操作的范围内"
        
        # 请求用户确认
        details = {
            "路径": path,
            "编辑指令": instructions
        }
        
        if not security_manager.confirm_action(
            message=f"即将编辑文件 '{path}'",
            risk_level=ToolRiskLevel.MEDIUM,
            details=details
        ):
            return "操作已取消"
        
        # 调用原始方法执行编辑
        return super().execute(path, instructions)


class SecureBash(Bash):
    """安全版Bash命令执行工具，集成了严格的安全检查"""
    
    def execute(self, command: str) -> str:
        # 检查工具是否被允许使用
        if not security_manager.is_tool_allowed("Bash", ToolRiskLevel.CRITICAL):
            return "安全错误: 当前安全级别下禁止执行Shell命令"
        
        # 请求用户确认（使用最高风险级别）
        details = {
            "命令": command
        }
        
        if not security_manager.confirm_action(
            message="即将执行Shell命令（高风险操作）",
            risk_level=ToolRiskLevel.CRITICAL,
            details=details
        ):
            return "操作已取消"
        
        # 调用原始方法执行命令
        return super().execute(command)


# 安全工具注册函数示例
def register_secure_tools(tool_registry):
    """将安全版工具注册到工具注册表中
    
    Args:
        tool_registry: 工具注册表实例
    """
    # 替换原始工具为安全版本
    tool_registry.register_tool(SecureFileWrite())
    tool_registry.register_tool(SecureFileEdit())
    tool_registry.register_tool(SecureBash())
