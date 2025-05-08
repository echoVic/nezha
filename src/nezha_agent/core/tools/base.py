"""
工具基类/接口
"""
from typing import Dict, Any

class BaseTool:
    name: str
    description: str
    arguments: Dict[str, Any]

    def execute(self, **kwargs):
        """执行工具的核心逻辑
        
        Args:
            **kwargs: 工具执行所需的参数
            
        Returns:
            Any: 工具执行结果
        """
        raise NotImplementedError()
