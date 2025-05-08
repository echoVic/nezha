"""功能层 - 提供核心功能实现

包含代码编辑、命令处理、Git集成、Shell集成和代理循环架构。
"""

from .agent.agent import NezhaAgent
from .commands.chat_command import ChatCommand
from .commands.plan_command import PlanCommand

__all__ = [
    'NezhaAgent',
    'ChatCommand',
    'PlanCommand',
]