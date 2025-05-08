"""集成层 - 提供外部集成支持

包含编辑器集成、Shell插件、外部工具集成和API服务。
"""

from .external.search import *
from .external.navigation import *
from .external.secure_tools import *

__all__ = [
    # 外部工具集成会在各自模块中定义
]