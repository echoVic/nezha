"""
工具基类/接口
"""
class BaseTool:
    name: str
    description: str
    arguments: dict

    def execute(self, **kwargs):
        raise NotImplementedError()
