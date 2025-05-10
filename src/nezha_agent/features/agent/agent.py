"""
Agent 核心逻辑类
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, Generator, List, Tuple

import json
import yaml

from ...core.context.context_engine import \
    ContextEngine  # Assuming ContextEngine provides context format
from ...core.models.llm_interface import LLMInterfaceBase, get_llm_interface
from ...core.security.security import SecurityManager
from ...core.tools.tool_registry import ToolRegistry
from ..function_call.function_call import FunctionCall
from ..function_call.handler import FunctionCallHandler

# 导入获取当前模型的函数
try:
    from ...ui.cli.cli import get_current_model
except ImportError:
    # 如果无法导入，定义一个空函数
    def get_current_model():
        return None

# 默认配置路径
DEFAULT_CONFIG_PATH = Path(os.path.expanduser("~/.config/nezha/config.yaml"))
# 默认内置配置路径
DEFAULT_BUILTIN_CONFIG_PATH = Path("config/default_config.yaml")

class NezhaAgent:
    def __init__(self, security_manager: SecurityManager, config_file: Optional[Path] = None, api_key: Optional[str] = None):
        self.security_manager = security_manager
        self.config_file = config_file
        self.config = self._load_config()
        self.api_key = api_key or os.environ.get("NEZHA_API_KEY")

        # 旧的初始化 FunctionCallHandler 已被移除
        # self.function_call_handler = FunctionCallHandler()

        # 优先使用内存中的模型设置
        current_model = get_current_model()
        if current_model:
            llm_config = dict(current_model)
            # 合并推理参数
            file_llm_config = self.config.get("llm", {})
            for key, value in file_llm_config.items():
                if key not in llm_config and key not in ["provider", "model", "api_key", "endpoint"]:
                    llm_config[key] = value
            print(f"\n使用内存中的模型设置: {current_model['name']}")
        else:
            # 读取 llm 字段和 models 列表
            llm_section = self.config.get("llm", {})
            model_id = llm_section.get("model")
            user_models = self.config.get("models", [])
            # 延迟导入 PREDEFINED_MODELS，避免循环引用
            try:
                from ...ui.cli.cli import PREDEFINED_MODELS
            except ImportError:
                PREDEFINED_MODELS = []
            all_models = PREDEFINED_MODELS + user_models
            # 查找完整模型配置
            model_conf = next((m for m in all_models if m.get("id") == model_id), None)
            llm_config = dict(model_conf) if model_conf else {}
            # 检查是否为预置模型且 api_key 需覆盖
            if model_conf in PREDEFINED_MODELS:
                # 仅当 api_key 为 **** 或空时才覆盖
                key_to_use = self.api_key or llm_section.get("api_key")
                if not llm_config.get("api_key") or llm_config.get("api_key") == "****":
                    if key_to_use and key_to_use != "****":
                        llm_config["api_key"] = key_to_use
                    else:
                        raise RuntimeError(
                            "预置模型未配置有效 api_key！请通过以下方式之一设置：\n"
                            "1. 命令行参数 --api-key 传入\n"
                            "2. 设置环境变量 NEZHA_API_KEY（如：export NEZHA_API_KEY=你的key）\n"
                            "3. 在 config.yaml 的 llm.api_key 字段填写专属 key（如：llm:\n  api_key: 你的key）\n"
                            "   编辑方法：\n"
                            "   - 终端输入 nano ~/.config/nezha/config.yaml （nano简单易用，Ctrl+O 保存，Ctrl+X 退出）\n"
                            "   - 或 vim ~/.config/nezha/config.yaml （高级用户）\n"
                            "   - 或 code ~/.config/nezha/config.yaml （如果已安装 VSCode）\n"
                            "如需试用请联系管理员获取 key。"
                        )
            # 确保 model 字段存在
            if "id" in llm_config and "model" not in llm_config:
                llm_config["model"] = llm_config["id"]
            # 合并推理参数（如 temperature/max_tokens）
            for key in ["temperature", "max_tokens", "verify_ssl"]:
                if key in llm_section:
                    llm_config[key] = llm_section[key]
        
        # 初始化 tool_choice, 从 llm_section 获取，默认为 "auto"
        self.tool_choice = llm_section.get("tool_choice", "auto")
        if self.tool_choice not in ["auto", "none"] and not (isinstance(self.tool_choice, dict) and self.tool_choice.get("type") == "function"):
            print(f"警告: 无效的 tool_choice 配置 '{self.tool_choice}'. 将使用 'auto'.")
            self.tool_choice = "auto"
                
        # 确保 verify_ssl 选项存在，默认为 True
        if "verify_ssl" not in llm_config:
            llm_config["verify_ssl"] = True
            
        # 初始化 LLM 接口
        try:
            self.llm_interface = get_llm_interface(llm_config)
            if not llm_config.get("verify_ssl", True):
                print("\n警告: SSL 证书验证已禁用。这可能会导致安全风险。")
        except Exception as e:
            raise RuntimeError(f"初始化 LLM 接口失败: {e}")

        self.history_manager = HistoryManager(config=self.config.get("history", {}))
        self.history = self.history_manager.load_history() # 加载历史记录

        self.context_engine = ContextEngine(self.llm_interface, config=self.config.get("context", {}))
        
        # Tool Registry initialization and population
        self.tool_registry = ToolRegistry()
        self._register_default_tools() # 注册默认工具
        self._register_plugin_tools()  # 注册插件工具

        # 初始化 FunctionCall 相关组件
        # 确保在 llm_interface 和 tool_registry 准备好之后进行
        self.function_call = FunctionCall(llm_interface=self.llm_interface)
        self.function_call_handler = FunctionCallHandler(
            function_call_instance=self.function_call,
            tool_registry=self.tool_registry
        )

        # TODO: Initialize other components like tool registry based on config

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件。如果找不到则报错，要求用户先执行 nezha init 初始化。"""
        # 首先尝试从配置文件加载
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                    if config:
                        print(f"\n成功从 {self.config_file} 加载配置")
                        return config
            except Exception as e:
                print(f"Error loading config file {self.config_file}: {e}")
                raise RuntimeError(f"加载配置文件失败: {e}")

        # 尝试从默认内置配置文件加载
        if DEFAULT_BUILTIN_CONFIG_PATH.exists():
            try:
                with open(DEFAULT_BUILTIN_CONFIG_PATH, 'r', encoding='utf-8') as f:
                    default_config = yaml.safe_load(f) or {}
                    if default_config:
                        print(f"\n成功从 {DEFAULT_BUILTIN_CONFIG_PATH} 加载默认配置")
                        return default_config
            except Exception as e:
                print(f"Error loading default config file {DEFAULT_BUILTIN_CONFIG_PATH}: {e}")
                raise RuntimeError(f"加载默认配置文件失败: {e}")

        # 没有任何可用配置，直接报错
        raise FileNotFoundError(
            f"未找到配置文件: {self.config_file or DEFAULT_CONFIG_PATH}，也未找到默认内置配置: {DEFAULT_BUILTIN_CONFIG_PATH}\n"
            f"请先运行 'nezha init' 进行初始化！"
        )

    def _execute_tools_and_get_final_response(self, 
                                            llm_initial_response_dict: Dict[str, Any], 
                                            history_leading_to_llm_call: List[Dict[str, str]],
                                            verbose: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
        """
        内部辅助方法：使用 FunctionCallHandler 执行工具调用并从LLM获取最终的非流式响应。
        
        Args:
            llm_initial_response_dict: LLM 首次返回的原始响应字典，其中包含工具调用请求。
            history_leading_to_llm_call: 导致上述 LLM 响应的对话历史。
            verbose: 是否打印详细日志。
            
        Returns:
            Tuple[str, List[Dict[str, Any]]]: 
                - 最终的文本响应字符串。
                - 包含从 history_leading_to_llm_call 开始，到最终文本响应为止的完整消息列表。
        """
        if verbose:
            print(f"\n[DEBUG] NezhaAgent._execute_tools_and_get_final_response starting...")
            print(f"  Initial LLM response (dict with tool calls): {str(llm_initial_response_dict)[:500]}...")
            print(f"  History leading to this LLM call: {history_leading_to_llm_call}")

        # --- Step 1: Handle the initial LLM response containing tool calls --- 
        # This executes tools and gets updated messages including tool results.
        handler_result_step1 = self.function_call_handler.handle_response(
            response=llm_initial_response_dict,
            original_messages=history_leading_to_llm_call
        )

        if verbose:
            print(f"  FunctionCallHandler result (step 1 - after tool execution): {str(handler_result_step1)[:500]}...")

        if handler_result_step1.get("type") == "text":
            if verbose:
                print(f"  [WARN] _execute_tools_and_get_final_response: Expected 'function_call' from handle_response in step 1, but got 'text'. Returning content.")
            # 如果第一步就是文本，那么 history_leading_to_llm_call 后面应该追加 llm_initial_response_dict['choices'][0]['message']
            # 但这不符合此函数的预期流程，所以直接返回，让调用者处理历史。
            # 或者，更健壮的做法是，也尝试从 llm_initial_response_dict 构建消息并返回
            # 为了简化，我们假设调用者会正确处理这种情况，或者这种情况不应该发生。
            final_text = handler_result_step1.get("content", "")
            # Try to reconstruct messages for this edge case
            updated_history = list(history_leading_to_llm_call)
            if llm_initial_response_dict.get("choices") and llm_initial_response_dict["choices"][0].get("message"):
                 updated_history.append(llm_initial_response_dict["choices"][0]["message"])
            return final_text, updated_history
        
        if handler_result_step1.get("type") != "function_call":
            if verbose:
                print(f"  [ERROR] _execute_tools_and_get_final_response: Unexpected type '{handler_result_step1.get('type')}' from handle_response in step 1.")
            return f"Error: Unexpected result type '{handler_result_step1.get('type')}' after initial tool processing.", history_leading_to_llm_call

        messages_after_tool_execution = handler_result_step1.get("messages")
        if not messages_after_tool_execution:
            if verbose:
                print(f"  [ERROR] _execute_tools_and_get_final_response: No messages returned by handle_response in step 1.")
            return "Error: Tool execution failed to produce subsequent messages for LLM.", history_leading_to_llm_call

        # --- Step 2: Call LLM again with tool results for a final text response --- 
        if verbose:
            print(f"  Preparing for final LLM call (tool_choice='none'). Messages being sent: {messages_after_tool_execution}")
        
        request_for_final_summary = self.function_call_handler.prepare_request(
            messages_after_tool_execution,
            model=self.llm_interface.model, # Assuming llm_interface has a .model attribute for the model ID
            tool_choice="none" # Crucial to prevent further tool call loops
        )

        if verbose:
            print(f"  Request data for final LLM call: {str(request_for_final_summary)[:500]}...")

        try:
            final_llm_response_dict = self.llm_interface.chat(
                **request_for_final_summary,
                stream=False # This helper method always gets a non-streamed final response
            )
            if verbose:
                print(f"  Raw final LLM response (dict from chat): {str(final_llm_response_dict)[:500]}...")
        except Exception as e:
            if verbose:
                print(f"  [ERROR] _execute_tools_and_get_final_response: Exception during final LLM call: {e}")
            # 返回错误和到目前为止的消息历史
            return f"Error during final LLM call after tool execution: {str(e)}", messages_after_tool_execution

        # --- Step 3: Handle the final LLM response, expecting text --- 
        handler_result_step2 = self.function_call_handler.handle_response(
            response=final_llm_response_dict,
            original_messages=messages_after_tool_execution # History that led to this final_llm_response_dict
        )

        if verbose:
            print(f"  FunctionCallHandler result (step 2 - after final LLM call): {str(handler_result_step2)[:500]}...")

        if handler_result_step2.get("type") == "text":
            final_content = handler_result_step2.get("content", "")
            final_messages_history = handler_result_step2.get("messages", messages_after_tool_execution)
            if verbose:
                print(f"  Extracted final content: {final_content[:200]}...")
                print(f"  Final message history from handler_result_step2: {final_messages_history}")
            return final_content, final_messages_history
        elif handler_result_step2.get("type") == "function_call":
            if verbose:
                print(f"  [ERROR] _execute_tools_and_get_final_response: Expected 'text' from handle_response in step 2, but got 'function_call' (LLM tried to call tools with tool_choice='none').")
            # 即使出错，也返回到目前为止的消息历史
            error_content = handler_result_step2.get("content", "Error: LLM unexpectedly called tools again after being instructed not to.")
            error_history = handler_result_step2.get("messages", messages_after_tool_execution)
            return error_content, error_history
        else:
            unknown_type = handler_result_step2.get('type')
            if verbose:
                print(f"  [ERROR] _execute_tools_and_get_final_response: Unknown handler result type '{unknown_type}' in step 2.")
            return f"Error: Unknown response type '{unknown_type}' after final LLM call.", messages_after_tool_execution

    def plan_chat(self, prompt: str, history: Optional[List[Dict[str, str]]] = None, 
                  stream: bool = False, verbose: bool = False) -> Union[str, Generator[str, None, None]]:
        """Handles the interactive planning chat loop.
        
        Args:
            prompt: The user's input prompt.
            history: The conversation history.
            stream: Whether to stream the response.
            verbose: Whether to print debug information.
            
        Returns:
            The LLM's response, either as a string or a generator of strings.
        """
        current_conversation_history = []

        # 1. 添加系统提示 (如果配置中存在)
        system_prompt_text = self.config.get("llm", {}).get("system_prompt")
        if system_prompt_text:
            current_conversation_history.append({"role": "system", "content": system_prompt_text})
            if verbose:
                # 打印部分系统提示，避免过长
                print(f"[DEBUG] Agent: Added system prompt: '{system_prompt_text[:100]}{'...' if len(system_prompt_text) > 100 else ''}'")

        # 2. 添加历史记录 (如果存在)
        if history:
            current_conversation_history.extend(history)
            if verbose:
                print(f"[DEBUG] Agent: Extended with {len(history)} messages from provided history.")

        # 3. 添加当前用户输入
        # 用户输入的内容应该已经是字符串了，由调用者（如ChatCommand）保证
        current_conversation_history.append({"role": "user", "content": prompt})
        if verbose:
            print(f"[DEBUG] Agent: Added current user prompt: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
            print(f"[DEBUG] Agent: History for first LLM call (total {len(current_conversation_history)} messages):")
            for i, msg in enumerate(current_conversation_history):
                content_display = str(msg.get('content', ''))
                print(f"  [{i}] Role: {msg.get('role')}, Content: '{content_display[:70]}{'...' if len(content_display) > 70 else ''}'")

        # 第一次调用LLM，判断是否需要工具或直接回复
        request_data = self.function_call_handler.prepare_request(
            current_conversation_history, # 使用构建好的完整历史
            model=self.llm_interface.model,
            tool_choice=self.tool_choice 
        )
        
        if verbose:
            print(f"\n[DEBUG] plan_chat - Request data for LLM: {{'model': '{request_data.get('model')}', 'num_messages': len(request_data.get('messages', [])), 'num_tools': len(request_data.get('tools', [])), 'tool_choice': request_data.get('tool_choice')}}")

        if not stream:
            # --- 非流式处理逻辑 ---
            llm_response_obj = self.llm_interface.chat(
                messages=request_data['messages'],
                tools=request_data.get('tools'),
                tool_choice=request_data.get('tool_choice'),
                stream=False
            )
            if verbose: print(f"\n[DEBUG] plan_chat (non-stream) - Raw LLM response obj: {str(llm_response_obj)[:500]}...")

            assistant_message_from_llm = None
            text_content_from_llm = ""

            # Normalize LLM response to extract assistant's message (potential tool_calls and content)
            if isinstance(llm_response_obj, str): # Should ideally be a structured dict
                assistant_message_from_llm = {"role": "assistant", "content": llm_response_obj, "tool_calls": None}
                text_content_from_llm = llm_response_obj
            elif isinstance(llm_response_obj, dict) and llm_response_obj.get("choices"):
                choices = llm_response_obj.get("choices", [])
                if choices and isinstance(choices[0], dict) and "message" in choices[0]:
                    message_data = choices[0]["message"]
                    if isinstance(message_data, dict): # Expected OpenAI format
                        assistant_message_from_llm = message_data
                        text_content_from_llm = message_data.get("content", "")
                    # Add other normalization if LLM returns message directly as str under 'message'
            else: # Fallback for unexpected structure
                assistant_message_from_llm = {"role": "assistant", "content": str(llm_response_obj), "tool_calls": None}
                text_content_from_llm = str(llm_response_obj)
            
            if verbose: print(f"\n[DEBUG] plan_chat (non-stream) - Extracted assistant message: {assistant_message_from_llm}")

            has_tool_calls = bool(assistant_message_from_llm and assistant_message_from_llm.get("tool_calls"))

            if has_tool_calls:
                if verbose: print(f"\n[DEBUG] plan_chat (non-stream) - Tool calls detected. Executing...")
                # Pass current_conversation_history as it contains the state up to the point LLM decided to call tools.
                # _execute_tools_and_get_final_response will append assistant_message_from_llm and tool results.
                final_text_reply, full_history_including_tools = self._execute_tools_and_get_final_response(
                    llm_response_obj, # Pass raw LLM response dict
                    current_conversation_history, # History up to the point LLM makes tool call decision
                    verbose=verbose
                )
                # 使用返回的完整历史更新代理的主历史记录
                self.history = full_history_including_tools # 或者 self.history.extend() / self.history_manager.add_messages()
                self.history_manager.save_history(self.history) # 保存更新后的历史
                return final_text_reply
            else:
                # 没有工具调用，直接是文本回复
                return text_content_from_llm or ""
            
        else:
            # --- 流式处理逻辑 ---
            def stream_processor() -> Generator[str, None, None]:
                # 变量用于累积流数据
                accumulated_content_parts = [] # 累积在工具调用信号前的文本内容
                tool_call_chunks = {} # key: tool_call_index, value: {id, name_parts, args_parts}
                
                has_started_tool_call_processing = False # 是否已开始处理工具调用块
                # Ensure history_for_tool_call is the state *before* LLM's response (assistant message with tool_calls)
                # This is current_conversation_history

                if verbose: print("\n[DEBUG] plan_chat (stream) - Starting stream processing...")

                llm_stream = self.llm_interface.chat(
                    messages=request_data['messages'],
                    tools=request_data.get('tools'),
                    tool_choice=request_data.get('tool_choice'),
                    stream=True
                )

                for chunk in llm_stream:
                    # 假设 chunk 是一个类似 OpenAI ChatCompletionChunk 的对象
                    # 它应该有 chunk.choices[0].delta.content 和 chunk.choices[0].delta.tool_calls
                    delta = None
                    if isinstance(chunk, dict) and chunk.get("choices") and isinstance(chunk["choices"], list) and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta")
                    
                    if not delta: 
                        # 如果块结构不符合预期，或为空，尝试将其作为字符串处理（如果适用）或跳过
                        if isinstance(chunk, str) and not has_started_tool_call_processing:
                            yield chunk # 直接输出，如果不是预期的对象格式
                        if verbose and not delta: print(f"[DEBUG] plan_chat (stream) - Skipping non-delta chunk: {str(chunk)[:100]}")
                        continue

                    chunk_text_content = delta.get("content")
                    chunk_tool_calls = delta.get("tool_calls")
                    
                    if verbose: print(f"[DEBUG] plan_chat (stream) - Chunk delta: {{'content': '{chunk_text_content}', 'tool_calls': {chunk_tool_calls}}}")

                    if chunk_tool_calls:
                        if not has_started_tool_call_processing:
                            has_started_tool_call_processing = True
                            # 输出之前累积的文本 (如果有)
                            if accumulated_content_parts:
                                yield "".join(accumulated_content_parts)
                                accumulated_content_parts = [] # 清空
                            yield "\n[INFO] 模型请求工具调用，正在处理中...\n" # 加载提示
                        
                        # 累积工具调用块
                        for tc_chunk in chunk_tool_calls:
                            index = tc_chunk.get("index")
                            if index is None: continue # 无效的工具调用块

                            if index not in tool_call_chunks:
                                tool_call_chunks[index] = {"id": None, "name_parts": [], "args_parts": [], "type": "function"}
                            
                            if tc_chunk.get("id"):
                                tool_call_chunks[index]["id"] = tc_chunk["id"]
                            if tc_chunk.get("function", {}).get("name"):
                                tool_call_chunks[index]["name_parts"].append(tc_chunk["function"]["name"])
                            if tc_chunk.get("function", {}).get("arguments"):
                                tool_call_chunks[index]["args_parts"].append(tc_chunk["function"]["arguments"])
                    
                    if chunk_text_content:
                        if has_started_tool_call_processing:
                            # 如果已开始处理工具调用，文本内容需要被缓冲，
                            # 因为它们可能属于LLM在决定调用工具后的思考过程，或与工具调用交错。
                            # 这个缓冲的文本将成为最终 assistant 消息的一部分。
                            accumulated_content_parts.append(chunk_text_content)
                        else:
                            # 在工具调用信号之前，直接输出文本
                            yield chunk_text_content
                    
                # --- 流结束后的处理 ---
                if verbose: print("\n[DEBUG] plan_chat (stream) - Stream finished.")

                if has_started_tool_call_processing and tool_call_chunks:
                    # 构建完整的 assistant 消息
                    final_tool_calls = []
                    for _index, tc_data in sorted(tool_call_chunks.items()):
                        final_tool_calls.append({
                            "id": tc_data["id"],
                            "type": tc_data["type"],
                            "function": {
                                "name": "".join(tc_data["name_parts"]),
                                "arguments": "".join(tc_data["args_parts"])
                            }
                        })
                    
                    assistant_message_for_tool_processing = {
                        "role": "assistant",
                        "content": "".join(accumulated_content_parts) if accumulated_content_parts else None,
                        "tool_calls": final_tool_calls
                    }
                    if verbose: print(f"[DEBUG] plan_chat (stream) - Constructed assistant message for tools: {assistant_message_for_tool_processing}")
                    
                    final_text_reply, full_history_including_tools = self._execute_tools_and_get_final_response(
                        {"choices": [{"message": assistant_message_for_tool_processing}]}, # Wrap in expected format
                        current_conversation_history, # History *before* this LLM's tool decision
                        verbose=verbose
                    )
                    yield final_text_reply
                elif accumulated_content_parts: # 如果有剩余的累积文本且没有工具调用
                    yield "".join(accumulated_content_parts)
                elif not has_started_tool_call_processing and not accumulated_content_parts and not tool_call_chunks: # 完全没有内容
                    if verbose: print("[DEBUG] plan_chat (stream) - Stream was empty and no tool calls.")
                    # yield "(No content from stream)" # 可以选择不输出任何东西
                    pass

            return stream_processor()

    def run(self, prompt: str, context: Optional[Dict[str, Any]] = None, verbose: bool = False, stream: bool = False):
        """执行用户指令
        
        Args:
            prompt: 用户输入的指令
            context: 上下文信息 (当前未使用，但保留以备将来扩展)
            verbose: 是否显示详细日志
            stream: 是否使用流式输出 (当前实现主要针对非流式，流式待完善)
            
        Returns:
            如果stream=True，返回生成器；否则返回字符串
        """
        
        # 检查是否是时间相关查询，如果是则直接调用工具
        time_keywords = ["几点", "几号", "时间", "日期", "星期", "现在", "今天", "当前", "几月"]
        
        # 打印所有关键词匹配情况
        print(f"\n[DEBUG] 检查时间关键词匹配情况:")
        for keyword in time_keywords:
            if keyword in prompt:
                print(f"[DEBUG] 关键词 '{keyword}' 匹配成功!")
            else:
                print(f"[DEBUG] 关键词 '{keyword}' 未匹配")
        
        # 强制匹配特定模式
        is_time_query = False
        if "几号" in prompt or "今天" in prompt:
            is_time_query = True
            print(f"\n[DEBUG] 强制匹配到时间查询模式!")
        
        if is_time_query or any(keyword in prompt for keyword in time_keywords):
            print(f"\n[DEBUG] 检测到时间相关查询: '{prompt}'，直接调用 get_current_time 工具")
            
            # 直接从工具注册表中获取工具并执行
            time_tool = self.function_call_handler.tool_registry.get("get_current_time")
            if time_tool:
                print(f"[DEBUG] 找到 get_current_time 工具，准备执行")
                try:
                    time_result = time_tool.execute()
                    print(f"[DEBUG] 工具执行成功，返回结果: {time_result}")
                    return time_result
                except Exception as e:
                    print(f"[ERROR] 执行 get_current_time 工具失败: {str(e)}")
            else:
                print(f"[ERROR] 无法找到 get_current_time 工具!")
        else:
            print(f"[DEBUG] 未检测到时间相关查询，继续正常处理流程")

        # 1. 准备消息历史
        # TODO: 从配置或更动态的来源加载系统提示
        # TODO: 如果有对话历史，需要在这里整合
        messages = [
            {"role": "system", "content": "你是 Nezha，一个乐于助人的AI助手。如果需要，你可以使用工具来回答问题。当被问到时间、日期相关问题时，必须使用get_current_time工具。"},
            {"role": "user", "content": prompt}
        ]

        if verbose and not stream:
            print(f"\n--- Sending Messages to LLM ---")
            for msg in messages:
                print(f"[{msg['role']}]: {msg['content']}")
            print("---------------------------------------")

        try:
            # 2. 准备 Function Call 请求
            request_data = self.function_call_handler.prepare_request(messages, model=self.llm_interface.model_id)
            
            # 3. 调用 LLM
            if verbose and not stream:
                print(f"\n--- Sending Request to LLM (with tools) ---")
                print(request_data)
                print("-------------------------------------------")
            
            raw_llm_response = self.llm_interface.chat(
                messages=request_data['messages'], 
                tools=request_data.get('tools'),
                tool_choice=request_data.get('tool_choice'),
                stream=False # Function call 流程目前不支持流式
            )

            if verbose and not stream:
                print(f"\n--- Raw LLM Response ---")
                print(raw_llm_response)
                print("------------------------")

            # 4. 处理 LLM 响应 (可能包含工具调用)
            # 打印原始响应以便调试
            if verbose:
                print(f"\n[DEBUG] 原始 LLM 响应: {raw_llm_response}")
            
            # 直接使用原始响应，不做额外转换
            # 如果是字符串，则将其包装为标准格式
            if isinstance(raw_llm_response, str):
                mock_openai_response = {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": raw_llm_response,
                            "tool_calls": None
                        }
                    }]
                }
            else:
                # 直接使用原始响应，假设它已经是正确的格式
                mock_openai_response = raw_llm_response
            
            # 检查是否有工具调用
            has_tool_calls = False
            tool_calls = None
            if isinstance(mock_openai_response, dict) and "choices" in mock_openai_response:
                choices = mock_openai_response.get("choices", [])
                if choices and isinstance(choices[0], dict) and "message" in choices[0]:
                    message = choices[0]["message"]
                    if isinstance(message, dict) and "tool_calls" in message and message["tool_calls"]:
                        has_tool_calls = True
                        tool_calls = message["tool_calls"]
                        if verbose:
                            print(f"\n[DEBUG] 检测到工具调用: {tool_calls}")
            
            # 如果有工具调用，直接执行工具（类似测试脚本的方式）
            if has_tool_calls and tool_calls:
                # 添加助手消息到历史中
                assistant_message = {
                    "role": "assistant",
                    "content": message.get("content", ""),
                    "tool_calls": tool_calls
                }
                messages.append(assistant_message)
                
                # 执行工具调用
                for tool_call in tool_calls:
                    if verbose:
                        print(f"\n[DEBUG] 执行工具: {tool_call.get('function', {}).get('name')}")
                    
                    # 获取工具名称和参数
                    tool_name = tool_call.get('function', {}).get('name')
                    arguments_str = tool_call.get('function', {}).get('arguments', "{}")
                    try:
                        arguments = json.loads(arguments_str)
                    except:
                        arguments = {}
                    
                    # 获取工具并执行
                    tool = self.function_call_handler.tool_registry.get(tool_name)
                    if tool:
                        try:
                            tool_result = tool.execute(**arguments)
                            if verbose:
                                print(f"[DEBUG] 工具返回结果: {tool_result}")
                            
                            # 添加工具响应到消息历史
                            tool_message = {
                                "role": "tool",
                                "tool_call_id": tool_call.get('id'),
                                "content": tool_result
                            }
                            messages.append(tool_message)
                        except Exception as e:
                            print(f"[ERROR] 执行工具失败: {e}")
                
                # 再次调用模型获取最终回复
                final_request_data = self.function_call_handler.prepare_request(
                    messages, 
                    model=self.llm_interface.model_id,
                    tool_choice="none"  # 防止无限循环调用工具
                )
                
                final_response = self.llm_interface.chat(
                    messages=final_request_data['messages'],
                    tools=final_request_data.get('tools'),
                    tool_choice=final_request_data.get('tool_choice'),
                    stream=False
                )
                
                # 提取最终回复内容
                if isinstance(final_response, str):
                    return final_response
                elif isinstance(final_response, dict) and "choices" in final_response:
                    return final_response["choices"][0]["message"].get("content", "")
                else:
                    return str(final_response)
            
            # 如果没有工具调用，直接提取文本内容返回
            if isinstance(mock_openai_response, dict) and "choices" in mock_openai_response:
                choices = mock_openai_response.get("choices", [])
                if choices and isinstance(choices[0], dict) and "message" in choices[0]:
                    message = choices[0]["message"]
                    if isinstance(message, dict) and "content" in message:
                        return message["content"]
            # 如果无法提取内容，返回原始响应
            return str(raw_llm_response)

            # 这里我们已经在上面直接处理了工具调用，不需要再使用 handler_result
            # 因此这里直接返回即可
            return "ERROR: 代码执行到了不应该执行的位置。请检查代码逻辑。"

        except Exception as e:
            error_msg = f"Error executing prompt: {e}"
            if verbose:
                import traceback
                print(f"Error during LLM call or function call handling: {e}\n{traceback.format_exc()}")
            
            if stream: # 虽然前面我们强制非流式，但以防万一
                def error_generator():
                    yield error_msg
                return error_generator()
            return error_msg