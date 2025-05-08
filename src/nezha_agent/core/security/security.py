"""安全层 - 集中处理危险操作的用户确认逻辑

该模块提供安全确认机制，用于在执行高风险操作前获取用户确认，
并根据配置的安全级别决定是否需要确认或完全禁用某些操作。
"""

import os
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.prompt import Confirm
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class SecurityLevel(Enum):
    """安全级别枚举
    
    - STRICT: 最严格模式，所有高风险操作都需要确认，某些操作可能被禁用
    - NORMAL: 标准模式，高风险操作需要确认
    - RELAXED: 宽松模式，只有极高风险操作需要确认
    - BYPASS: 跳过所有确认（危险！仅用于自动化脚本）
    """
    STRICT = auto()
    NORMAL = auto()
    RELAXED = auto()
    BYPASS = auto()


class ToolRiskLevel(Enum):
    """工具风险级别枚举
    
    - NONE: 无风险，如只读操作
    - LOW: 低风险，如创建新文件
    - MEDIUM: 中等风险，如修改文件
    - HIGH: 高风险，如删除文件
    - CRITICAL: 极高风险，如执行Shell命令
    """
    NONE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class SecurityManager:
    """安全管理器，处理所有安全相关的逻辑"""
    
    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.NORMAL,
        allowed_paths: Optional[List[str]] = None,
        disabled_tools: Optional[List[str]] = None,
        yes_to_all: bool = False
    ):
        """初始化安全管理器
        
        Args:
            security_level: 安全级别
            allowed_paths: 允许操作的路径列表，None表示不限制
            disabled_tools: 禁用的工具列表
            yes_to_all: 是否对所有确认自动回答是（危险！）
        """
        self.security_level = security_level
        self.allowed_paths = [Path(p).expanduser().resolve() for p in (allowed_paths or [])]
        self.disabled_tools = disabled_tools or []
        self.yes_to_all = yes_to_all
        self.console = Console() if RICH_AVAILABLE else None
        
        # 各安全级别下需要确认的工具风险等级
        self._confirm_thresholds = {
            SecurityLevel.STRICT: ToolRiskLevel.LOW,      # 严格模式：低风险及以上需确认
            SecurityLevel.NORMAL: ToolRiskLevel.MEDIUM,  # 标准模式：中风险及以上需确认
            SecurityLevel.RELAXED: ToolRiskLevel.HIGH,   # 宽松模式：高风险及以上需确认
            SecurityLevel.BYPASS: ToolRiskLevel.CRITICAL # 跳过模式：仅极高风险需确认
        }
        
        # 各安全级别下禁用的工具风险等级
        self._disable_thresholds = {
            SecurityLevel.STRICT: ToolRiskLevel.CRITICAL,  # 严格模式：禁用极高风险
            SecurityLevel.NORMAL: None,                  # 标准模式：不自动禁用
            SecurityLevel.RELAXED: None,                 # 宽松模式：不自动禁用
            SecurityLevel.BYPASS: None                   # 跳过模式：不自动禁用
        }
    
    def is_tool_allowed(self, tool_name: str, risk_level: ToolRiskLevel) -> bool:
        """检查工具是否被允许使用
        
        Args:
            tool_name: 工具名称
            risk_level: 工具风险级别
            
        Returns:
            bool: 是否允许使用该工具
        """
        # 检查工具是否在禁用列表中
        if tool_name in self.disabled_tools:
            return False
            
        # 检查工具风险级别是否超过当前安全级别的禁用阈值
        disable_threshold = self._disable_thresholds[self.security_level]
        if disable_threshold and risk_level.value >= disable_threshold.value:
            return False
            
        return True
    
    def is_path_allowed(self, path: str) -> bool:
        """检查路径是否在允许的范围内
        
        Args:
            path: 要检查的路径
            
        Returns:
            bool: 如果路径在允许范围内返回True，否则返回False
        """
        # 如果没有设置允许路径，则所有路径都允许
        if not self.allowed_paths:
            return True
            
        # 将路径转换为绝对路径
        abs_path = Path(path).expanduser().resolve()
        
        # 检查路径是否在允许的路径列表中
        for allowed_path in self.allowed_paths:
            if abs_path == allowed_path or allowed_path in abs_path.parents:
                return True
                
        return False
    
    def confirm_action(
        self,
        message: str,
        risk_level: ToolRiskLevel,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """请求用户确认操作
        
        Args:
            message: 确认消息
            risk_level: 操作风险级别
            details: 操作详情，用于展示给用户
            
        Returns:
            bool: 用户是否确认执行操作
        """
        # 如果设置了yes_to_all，直接返回True
        if self.yes_to_all:
            return True
            
        # 检查当前安全级别下是否需要确认该风险级别的操作
        confirm_threshold = self._confirm_thresholds[self.security_level]
        if risk_level.value < confirm_threshold.value:
            return True  # 风险级别低于阈值，无需确认
        
        # 格式化详情信息
        detail_str = ""
        if details:
            detail_str = "\n详情:\n" + "\n".join(f"  {k}: {v}" for k, v in details.items())
        
        # 使用rich库进行美化确认（如果可用）
        if RICH_AVAILABLE:
            self.console.print(f"[bold yellow]安全确认[/bold yellow]: {message}")
            if detail_str:
                self.console.print(detail_str)
            return Confirm.ask("是否继续?")
        else:
            # 回退到基本确认方式
            print(f"安全确认: {message}{detail_str}")
            resp = input("是否继续? [y/N]: ").strip().lower()
            return resp == "y"


# 默认安全管理器实例
security_manager = SecurityManager()


# 向后兼容的函数
def confirm_action(message: str) -> bool:
    """请求用户确认操作（向后兼容）
    
    Args:
        message: 确认消息
        
    Returns:
        bool: 用户是否确认执行操作
    """
    return security_manager.confirm_action(
        message=message,
        risk_level=ToolRiskLevel.MEDIUM
    )