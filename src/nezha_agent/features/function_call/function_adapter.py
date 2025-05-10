"""
工具到 Function Call 格式的适配器
"""
from typing import Dict, List, Any, Type, Optional
from ...core.tools.base import BaseTool

def tool_to_function_schema(tool: BaseTool) -> Dict[str, Any]:
    """
    将工具对象转换为 Function Call 格式的 schema
    
    Args:
        tool: 工具对象
        
    Returns:
        Dict: Function Call 格式的工具定义
    """
    # 创建基本的schema结构
    schema = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
        }
    }
    
    # 无论是否有参数，都添加 parameters 字段
    # 这是为了与千问的API格式保持一致
    parameters_schema = {
        "type": "object",
        "properties": {},
    }
    
    required_params_list = []
    
    # 如果有参数定义，处理参数
    if hasattr(tool, 'arguments') and tool.arguments:
        for arg_name, arg_config in tool.arguments.items():
            param_details = {}
            # 简单字符串描述的参数 (视为必选的字符串类型)
            if isinstance(arg_config, str):
                param_details = {
                    "type": "string",
                    "description": arg_config
                }
                if arg_name not in required_params_list: # 避免重复添加（理论上不应发生）
                    required_params_list.append(arg_name)
            # 字典形式的参数定义
            elif isinstance(arg_config, dict):
                # Type (支持 'int' 作为 'integer' 的别名, 默认为 'string')
                # 支持的 JSON Schema 类型: string, integer, number, boolean, array, object
                param_type = arg_config.get("type", "string")
                if param_type == "int":
                    param_details["type"] = "integer"
                else:
                    param_details["type"] = param_type
                
                # Description
                param_details["description"] = arg_config.get("desc", arg_config.get("description", "")) # 兼容 desc 和 description
                
                # Enum (通常用于 string 类型)
                if "enum" in arg_config and isinstance(arg_config["enum"], list):
                    param_details["enum"] = arg_config["enum"]
                
                # Items (用于 array 类型)
                if param_details["type"] == "array" and "items" in arg_config and isinstance(arg_config["items"], dict):
                    param_details["items"] = arg_config["items"]
                
                # Properties (用于 object 类型, 可选的初步支持)
                if param_details["type"] == "object" and "properties" in arg_config and isinstance(arg_config["properties"], dict):
                    param_details["properties"] = arg_config["properties"]
                    # 如果 object 本身有内部 required 字段，也可以处理
                    if "required" in arg_config and isinstance(arg_config["required"], list):
                        param_details["required"] = arg_config["required"]
                
                # 检查参数是否必需 (默认为 True)
                if arg_config.get("required", True) and arg_name not in required_params_list:
                    required_params_list.append(arg_name)
            else:
                # 跳过无法解析的参数配置
                print(f"[WARNING] 工具 '{tool.name}' 的参数 '{arg_name}' 配置格式无法识别: {arg_config}")
                continue
            
            parameters_schema["properties"][arg_name] = param_details
        
    # 只有在有必需参数时才添加 required 字段到 parameters_schema
    if required_params_list:
        parameters_schema["required"] = required_params_list
    
    # 始终添加 parameters 字段到 function schema，即使它是空的 properties
    schema["function"]["parameters"] = parameters_schema
    
    print(f"[DEBUG] 为工具 {tool.name} 生成的schema: {schema}")
    return schema

def convert_tools_to_functions(tools: List[BaseTool]) -> List[Dict[str, Any]]:
    """
    批量转换工具为 Function Call 格式
    
    Args:
        tools: 工具对象列表
        
    Returns:
        List[Dict]: Function Call 格式的工具定义列表
    """
    return [tool_to_function_schema(tool) for tool in tools]

def execute_function_call(tool_registry, function_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行 Function Call 调用
    
    Args:
        tool_registry: 工具注册表
        function_call: Function Call 调用信息
        
    Returns:
        Dict: 执行结果
    """
    tool_name = function_call.get("name")
    arguments = function_call.get("arguments", {})
    
    tool = tool_registry.get(tool_name)
    if not tool:
        return {
            "id": function_call.get("id"),
            "content": f"工具 {tool_name} 未注册",
            "error": "TOOL_NOT_FOUND"
        }
    
    try:
        result = tool.execute(**arguments)
        return {
            "id": function_call.get("id"),
            "content": str(result),
            "error": None
        }
    except Exception as e:
        return {
            "id": function_call.get("id"),
            "content": f"工具执行失败: {str(e)}",
            "error": "EXECUTION_ERROR"
        }
