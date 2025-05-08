"""
nezha CLI 入口
"""
import sys

# 直接导入 cli 模块
from src.nezha_agent.cli import app

if __name__ == "__main__":
    sys.exit(app())
