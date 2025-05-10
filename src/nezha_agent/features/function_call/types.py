"""
Function Call 类型定义
"""
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class FunctionSchema:
    """函数调用模式定义"""
    name: str
    description: str
    parameters: Optional[Dict[str, Any]] = None
    required: Optional[List[str]] = None


@dataclass
class FunctionCallInfo:
    """函数调用信息"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """工具执行结果"""
    id: str
    content: str
    error: Optional[str] = None
