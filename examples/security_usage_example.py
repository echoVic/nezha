"""\nnezha 安全层使用示例\n\n本示例展示如何在实际应用中初始化和使用安全管理器。\n"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from nezha.security import (SecurityLevel, SecurityManager, ToolRiskLevel,
                            security_manager)


def demo_default_security():
    """演示使用默认安全管理器"""
    print("\n=== 使用默认安全管理器 ===")
    
    # 默认安全管理器使用NORMAL安全级别
    print(f"当前安全级别: {security_manager.security_level}")
    
    # 检查工具是否允许使用
    print(f"FileRead工具允许使用: {security_manager.is_tool_allowed('FileRead', ToolRiskLevel.NONE)}")
    print(f"Bash工具允许使用: {security_manager.is_tool_allowed('Bash', ToolRiskLevel.CRITICAL)}")
    
    # 检查路径是否允许操作（默认不限制路径）
    print(f"当前目录允许操作: {security_manager.is_path_allowed(os.getcwd())}")
    
    # 模拟确认操作（在实际运行时会请求用户输入）
    print("模拟低风险操作确认（在NORMAL级别下无需确认）:")
    # 实际运行时会返回True，因为LOW风险在NORMAL级别下无需确认
    security_manager.confirm_action(
        message="执行低风险操作",
        risk_level=ToolRiskLevel.LOW
    )
    
    print("\n模拟高风险操作确认（在NORMAL级别下需要确认）:")
    # 实际运行时会请求用户确认
    security_manager.confirm_action(
        message="执行高风险操作",
        risk_level=ToolRiskLevel.HIGH,
        details={"操作": "删除文件", "路径": "/path/to/file"}
    )


def demo_custom_security():
    """演示使用自定义安全管理器"""
    print("\n=== 使用自定义安全管理器 ===")
    
    # 创建自定义安全管理器
    custom_security = SecurityManager(
        security_level=SecurityLevel.STRICT,
        allowed_paths=[os.path.expanduser("~/Documents")],
        disabled_tools=["Bash"],
        yes_to_all=False
    )
    
    print(f"当前安全级别: {custom_security.security_level}")
    
    # 检查工具是否允许使用
    print(f"FileRead工具允许使用: {custom_security.is_tool_allowed('FileRead', ToolRiskLevel.NONE)}")
    print(f"Bash工具允许使用: {custom_security.is_tool_allowed('Bash', ToolRiskLevel.CRITICAL)}")
    
    # 检查路径是否允许操作
    home_docs = os.path.expanduser("~/Documents")
    random_path = "/tmp/random"
    print(f"~/Documents目录允许操作: {custom_security.is_path_allowed(home_docs)}")
    print(f"/tmp目录允许操作: {custom_security.is_path_allowed(random_path)}")
    
    # 模拟确认操作
    print("\n在STRICT级别下，低风险操作也需要确认:")
    custom_security.confirm_action(
        message="执行低风险操作",
        risk_level=ToolRiskLevel.LOW
    )


def main():
    """主函数"""
    print("nezha 安全层使用示例")
    
    demo_default_security()
    demo_custom_security()
    
    print("\n示例结束")


if __name__ == "__main__":
    main()