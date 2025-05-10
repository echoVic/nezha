"""
Function Call 处理器
"""
import json
from typing import Dict, List, Any, Optional, Union
from ...core.tools.tool_registry import ToolRegistry
from .function_call import FunctionCall
from .function_adapter import execute_function_call

class FunctionCallHandler:
    """
    Function Call 处理器，用于处理 Nezha 模型的 Function Call 请求和响应
    """
    def __init__(self):
        self.function_call = FunctionCall()
        self.tool_registry = ToolRegistry()
        
        # 自动注册所有工具
        self.register_all_tools()
        print(f"[DEBUG] FunctionCallHandler 初始化完成，已注册 {len(self.tool_registry.get_all_tools())} 个工具")
        
    def register_all_tools(self):
        """注册所有工具到 Function Call 系统"""
        tool_schemas = self.tool_registry.get_function_schemas()
        self.function_call.register_tools(tool_schemas)
        for tool in self.tool_registry.get_all_tools():
            print(f"[DEBUG] 已注册工具: {tool.name} - {tool.description}")
    
    def prepare_request(self, 
                        messages: List[Dict[str, str]], 
                        model: str = "nezha-base",
                        tool_choice: Union[str, Dict[str, Any]] = "auto") -> Dict[str, Any]:
        """
        准备 Function Call 请求
        
        Args:
            messages: 对话历史
            model: 使用的模型名称
            tool_choice: 工具选择策略
            
        Returns:
            Dict: 格式化的请求数据
        """
        # 处理最后一条消息，如果含有时间相关的关键词，增强提示
        if messages and isinstance(messages, list) and len(messages) > 0:
            last_msg = messages[-1]
            if last_msg.get("role") == "user":
                content_value = last_msg.get("content", "") # Can be str or list from previous steps

                # 确保用户消息的 content 字段是符合多部分消息格式的列表
                if isinstance(content_value, str):
                    print(f"[DEBUG] 标准化用户消息的字符串 content 为列表格式: '{content_value}'")
                    last_msg["content"] = [{"type": "text", "text": content_value}]
                    content_value = last_msg["content"] # 更新 content_value 以引用新的列表
                elif isinstance(content_value, list):
                    # 如果已经是列表，确保其内部元素是正确的字典结构
                    for i, item in enumerate(content_value):
                        if isinstance(item, str):
                            print(f"[DEBUG] 标准化用户消息 content 列表中的字符串元素: '{item}'")
                            content_value[i] = {"type": "text", "text": item}
                        elif not (isinstance(item, dict) and "type" in item and "text" in item):
                            # 对于其他不规范的结构，尝试转换为文本，或记录警告
                            print(f"[WARN] 用户消息 content 列表中发现不规范元素，将尝试转换为文本: {item}")
                            content_value[i] = {"type": "text", "text": str(item)}
                else:
                    # 对于其他意外的 content 类型，记录警告并尝试转换为标准格式
                    print(f"[WARN] 用户消息 content 具有意外类型 '{type(content_value)}'，将尝试转换为标准文本格式: {content_value}")
                    last_msg["content"] = [{"type": "text", "text": str(content_value)}]
                    content_value = last_msg["content"]

                time_keywords = ["几点", "几号", "时间", "日期", "星期", "现在", "今天", "当前"]
                
                text_to_check_for_keywords = ""
                # 从 content_value (现在应该是列表) 中提取文本内容进行关键词检测
                if isinstance(content_value, list):
                    for item in content_value:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_to_check_for_keywords += item.get("text", "") + " " # 将所有文本部分连接起来
                text_to_check_for_keywords = text_to_check_for_keywords.strip()

                if any(keyword in text_to_check_for_keywords for keyword in time_keywords):
                    print(f"[DEBUG] 检测到时间相关关键词，用户输入 (合并后): {text_to_check_for_keywords}")
                    # enhanced_prompt_suffix = "\n[Nezha Notes: Query contains time-sensitive keywords. Consider using 'get_current_time' tool to get precise information.]"
                    # if isinstance(last_msg["content"], list): # It should be a list now
                    #     last_msg["content"].append({"type": "text", "text": enhanced_prompt_suffix})
                    #     print(f"[DEBUG] 已追加增强用户提示到列表 (逻辑已注释): {last_msg['content']}")

        # 调试打印，查看修改后的 messages 结构
        if messages and isinstance(messages, list) and len(messages) > 0:
            print(f"[DEEP_DEBUG] FunctionCallHandler - messages[0] after standardization: {messages[0]}")

        request = self.function_call.format_request(messages, model, tool_choice)
        print(f"[DEBUG] 准备 Function Call 请求: \n  - 模型: {model}\n  - 工具选择: {tool_choice}\n  - 工具数量: {len(request.get('tools', []))}")
        return request
    
    def handle_response(self, 
                        response: Dict[str, Any], 
                        original_messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        处理模型返回的 Function Call 响应
        
        Args:
            response: 模型响应 (原始字典格式从 LLM API)
            original_messages: 原始对话历史
            
        Returns:
            Dict: 处理结果，包含最终回复或下一步请求
        """
        print(f"[DEBUG] 收到模型响应: {response}")
        
        # 解析响应以供工具执行，这部分逻辑保持不变
        parsed_for_execution = self.function_call.parse_response(response)
        print(f"[DEBUG] 解析后的响应 (供执行): {parsed_for_execution}")
        
        # 检查是否有工具调用 (根据解析结果)
        if not parsed_for_execution.get("tool_calls"):
            print(f"[DEBUG] 没有检测到工具调用，返回文本响应")
            return {
                "type": "text",
                "content": parsed_for_execution.get("content", ""), # 使用解析后的 content
                "messages": original_messages # 原始消息，因为没有发生交互
            }
        
        # 执行工具调用 (使用解析后的 tool_calls)
        print(f"[DEBUG] 检测到 {len(parsed_for_execution.get('tool_calls', []))} 个工具调用，准备执行")
        tool_results = []
        for tool_call_for_exec in parsed_for_execution.get("tool_calls", []):
            print(f"[DEBUG] 执行工具: {tool_call_for_exec.get('name')} 参数: {tool_call_for_exec.get('arguments')}")
            # 调用 execute_function_call 时，它需要的是解析后的 name 和 arguments
            result = execute_function_call(self.tool_registry, tool_call_for_exec)
            print(f"[DEBUG] 工具返回结果: {result}")
            tool_results.append(result)
        
        # 准备下一次请求或添加到历史的消息
        # IMPORTANT: 当将 assistant 消息添加到历史记录时，tool_calls 必须是LLM原始返回的格式
        # 而不是 self.function_call.parse_response() 返回的、已解析的格式。
        raw_llm_message = response.get("choices", [{}])[0].get("message", {})
        raw_tool_calls_from_llm = raw_llm_message.get("tool_calls") # 这应该是LLM原始的tool_calls列表

        assistant_message = {
            "role": "assistant",
            "content": parsed_for_execution.get("content"), # content 可以来自解析结果
            "tool_calls": raw_tool_calls_from_llm # 使用LLM原始的tool_calls
        }
        
        # 确保 assistant_message 中的 tool_calls 是列表，即使只有一个
        if assistant_message["tool_calls"] and not isinstance(assistant_message["tool_calls"], list):
             print(f"[WARN] raw_tool_calls_from_llm 不是列表，尝试包装: {assistant_message['tool_calls']}")
             assistant_message["tool_calls"] = [assistant_message["tool_calls"]]
        elif not assistant_message["tool_calls"]:
             # 如果原始响应中没有 tool_calls（理论上不应进入此分支，因为前面有检查），
             # 但 parsed_for_execution.get("tool_calls") 有，则日志警告，并清空
             if parsed_for_execution.get("tool_calls"):
                  print(f"[WARN] 逻辑不一致：parsed_for_execution 有 tool_calls 但 raw_llm_message 没有。清空 assistant_message.tool_calls")
             assistant_message["tool_calls"] = [] # 或者 None，取决于LLM接受什么

        updated_messages = self.function_call.format_tool_response(
            original_messages, 
            assistant_message, 
            tool_results
        )
        
        print(f"[DEBUG] 已格式化工具调用响应，消息数: {len(updated_messages)}")
        if updated_messages and len(updated_messages) > 1:
            print(f"[DEEP_DEBUG] assistant_message in history: {updated_messages[-2]}") # 假设工具结果是最后一个

        return {
            "type": "function_call",
            "tool_results": tool_results,
            "messages": updated_messages
        }
    
    def complete_function_call(self, 
                              response: Dict[str, Any], 
                              original_messages: List[Dict[str, str]],
                              model: str = "nezha-base") -> Dict[str, Any]:
        """
        完成整个 Function Call 流程
        
        Args:
            response: 模型响应
            original_messages: 原始对话历史
            model: 使用的模型名称
            
        Returns:
            Dict: 最终处理结果
        """
        # 处理响应
        result = self.handle_response(response, original_messages)
        
        # 如果不是工具调用，直接返回
        if result.get("type") != "function_call":
            print(f"[DEBUG] 非工具调用响应，直接返回")
            return result
        
        # 准备下一次请求
        print(f"[DEBUG] 准备工具调用后的最终请求（tool_choice=none）")
        next_request = self.prepare_request(
            result.get("messages", []),
            model=model,
            tool_choice="none"  # 防止无限循环调用工具
        )
        
        return {
            "type": "next_request",
            "request": next_request,
            "tool_results": result.get("tool_results", []),
            "messages": result.get("messages", [])
        }
