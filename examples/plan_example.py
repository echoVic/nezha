#!/usr/bin/env python
"""
交互式规划命令示例

这个示例展示了如何使用nezha的plan命令进行交互式需求规划。

使用方法:
    python examples/plan_example.py
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import NezhaAgent
from context_engine import ContextEngine
from plan_command import PlanCommand
from security import SecurityLevel, SecurityManager


def main():
    """运行plan命令示例"""
    # 设置工作目录为项目根目录
    os.chdir(Path(__file__).parent.parent)
    
    # 初始化组件
    security_manager = SecurityManager(SecurityLevel.NORMAL)
    context_engine = ContextEngine(working_dir=os.getcwd())
    agent = NezhaAgent(security_manager=security_manager)
    
    # 初始化规划命令
    planner = PlanCommand(
        agent=agent,
        context_engine=context_engine,
        verbose=True,
        output_file=Path("plan_output.md")
    )
    
    # 执行交互式规划
    initial_requirement = "为项目添加单元测试框架和CI/CD集成"
    planner.run(initial_requirement)

if __name__ == "__main__":
    main()