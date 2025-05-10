"""
Function Call 功能实现
"""
import json
from typing import Dict, List, Any, Optional, Union

class FunctionCall:
    """
    Function Call 功能的核心实现类
    """
    def __init__(self):
        self.tools = []
        
    def register_tool(self, tool_schema: Dict[str, Any]) -> None:
        """
        注册工具到 Function Call 系统
        
        Args:
            tool_schema: 符合 OpenAI 格式的工具定义
        """
        self.tools.append(tool_schema)
    
    def register_tools(self, tool_schemas: List[Dict[str, Any]]) -> None:
        """
        批量注册工具到 Function Call 系统
        
        Args:
            tool_schemas: 符合 OpenAI 格式的工具定义列表
        """
        self.tools.extend(tool_schemas)
    
    def format_request(self, 
                       messages: List[Dict[str, str]], 
                       model: str = "nezha-base",
                       tool_choice: Union[str, Dict[str, Any]] = "auto") -> Dict[str, Any]:
        """
        格式化 Function Call 请求
        
        Args:
            messages: 对话历史
            model: 使用的模型名称
            tool_choice: 工具选择策略，可以是 "auto", "none" 或指定工具
            
        Returns:
            Dict: 格式化的请求数据
        """
        request = {
            "model": model,
            "messages": messages,
            "tools": self.tools
        }
        
        if tool_choice:
            request["tool_choice"] = tool_choice
            
        return request
    
    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析模型返回的 Function Call 响应
        
        Args:
            response: 模型响应
            
        Returns:
            Dict: 解析后的响应数据，包含工具调用信息
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                print("[ERROR] LLM 响应中 choices 列表为空或不存在")
                return {"error": "无效的响应格式: choices为空或不存在", "content": None, "tool_calls": []}
                
            message = choices[0].get("message", {})
            if not message:
                print("[ERROR] LLM 响应的 choices[0] 中 message 为空或不存在")
                return {"error": "无效的响应格式: message为空或不存在", "content": None, "tool_calls": []}
            
            tool_calls_raw = message.get("tool_calls", [])
            
            result = {
                "content": message.get("content"), # 可能为 None，如果只有工具调用
                "tool_calls": []
            }
            
            if not tool_calls_raw:
                # 没有工具调用，可能是纯文本回复
                print("[DEBUG] LLM 响应中没有 tool_calls")
                return result
                
            for tool_call_item in tool_calls_raw:
                if tool_call_item.get("type") == "function":
                    function_data = tool_call_item.get("function", {})
                    tool_name = function_data.get("name")
                    tool_id = tool_call_item.get("id")
                    
                    if not tool_name or not tool_id:
                        print(f"[WARNING] 跳过格式不正确的工具调用项: {tool_call_item}")
                        continue
                        
                    parsed_arguments = {}
                    raw_arguments = function_data.get("arguments")
                    
                    if isinstance(raw_arguments, dict):
                        parsed_arguments = raw_arguments
                        print(f"[DEBUG] 工具 '{tool_name}' (ID: {tool_id}) 的参数已是字典格式: {parsed_arguments}")
                    elif isinstance(raw_arguments, str):
                        try:
                            parsed_arguments = json.loads(raw_arguments)
                            print(f"[DEBUG] 工具 '{tool_name}' (ID: {tool_id}) 的参数从JSON字符串解析: {parsed_arguments}")
                        except json.JSONDecodeError as e:
                            print(f"[WARNING] 工具 '{tool_name}' (ID: {tool_id}) 的参数JSON解析失败: {e}. 原始参数: '{raw_arguments}'")
                            # 保留 parsed_arguments 为空字典
                    elif raw_arguments is None:
                        print(f"[DEBUG] 工具 '{tool_name}' (ID: {tool_id}) 没有提供参数 (arguments is None)")
                        # 保留 parsed_arguments 为空字典
                    else:
                        print(f"[WARNING] 工具 '{tool_name}' (ID: {tool_id}) 的参数类型未知: {type(raw_arguments)}. 原始参数: '{raw_arguments}'")
                        # 保留 parsed_arguments 为空字典
                                            
                    result["tool_calls"].append({
                        "id": tool_id,
                        "name": tool_name,
                        "arguments": parsed_arguments
                    })
                else:
                    print(f"[WARNING] 跳过未知类型的工具调用项: {tool_call_item.get('type')}")
            
            print(f"[DEBUG] 解析后的响应 tool_calls: {result['tool_calls']}")
            return result
            
        except Exception as e:
            print(f"[ERROR] 解析LLM响应时发生意外错误: {str(e)}")
            return {"error": f"解析响应失败: {str(e)}", "content": None, "tool_calls": []}
    
    def format_tool_response(self, 
                            original_messages: List[Dict[str, str]], 
                            assistant_message: Dict[str, Any],
                            tool_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        格式化工具执行结果，准备下一次请求
        
        Args:
            original_messages: 原始对话历史
            assistant_message: 助手消息（包含工具调用）
            tool_results: 工具执行结果列表
            
        Returns:
            List: 更新后的消息列表
        """
        messages = original_messages.copy()
        messages.append(assistant_message)
        
        for result in tool_results:
            tool_message = {
                "role": "tool",
                "tool_call_id": result.get("id"),
                "content": result.get("content", "")
            }
            messages.append(tool_message)
            
        return messages
