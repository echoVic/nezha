"""命令行接口模块

提供命令行工具和交互界面。
"""

from .cli import app, get_current_model

__all__ = [
    'app',
    'get_current_model',
]