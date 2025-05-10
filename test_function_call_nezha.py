from openai import OpenAI
from datetime import datetime
import json
import os
import random
from typing import Union, Dict, Any

# 使用我们自己的函数调用实现
from nezha_agent.features.function_call.handler import FunctionCallHandler
from nezha_agent.features.tools.weather_time import GetCurrentTime, GetCurrentWeather

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv("NEZHA_API_KEY"),  # 使用NEZHA_API_KEY环境变量
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope SDK的base_url
)

# 初始化函数调用处理器
function_handler = FunctionCallHandler()

# 封装模型响应函数
def get_response(messages, tool_choice: Union[str, Dict[str, Any]] = "auto"):
    print(f"发送消息到模型 (tool_choice: {tool_choice}): {messages}")
    
    # 准备请求数据
    request_data = function_handler.prepare_request(messages, model="qwen-plus", tool_choice=tool_choice)
    
    # 打印请求数据中的工具信息
    print(f"工具数量: {len(request_data.get('tools', []))}")
    
    # 调用模型
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=request_data.get('messages'), # 使用prepare_request处理过的messages
        tools=request_data.get('tools'),
        tool_choice=request_data.get('tool_choice')
    )
    
    print(f"收到模型响应: {completion}")
    return completion.model_dump() # 返回字典格式，方便后续处理

def call_with_messages():
    print("\n")
    messages = [
        {
            "content": input("请输入："),  # 提问示例："现在几点了？" "北京天气如何？"
            "role": "user",
        }
    ]
    print("-" * 60)
    
    current_messages = messages
    max_turns = 5 # 防止无限循环

    for turn in range(1, max_turns + 1):
        print(f"\n--- 第 {turn} 轮对话 ---")
        
        # 第一轮调用，或者工具调用后的下一轮（非强制none）
        tool_choice_for_this_turn = "auto"
        if turn > 1: # 如果是工具调用后的回应，则不再主动要求工具
             # 这个逻辑会被下面的 handle_response 结果覆盖，如果需要再次调用LLM
             pass 
        
        llm_response_dict = get_response(current_messages, tool_choice=tool_choice_for_this_turn)
        
        # 使用 FunctionCallHandler 处理响应
        # 注意：handle_response 期望的 original_messages 是调用LLM之前的消息列表
        handler_result = function_handler.handle_response(llm_response_dict, current_messages)
        
        print(f"[DEBUG] FunctionCallHandler.handle_response 结果: {handler_result}")

        if handler_result.get("type") == "text":
            final_answer = handler_result.get("content", "")
            print(f"\n最终答案 (来自 handle_response): {final_answer}")
            break # 结束对话
        elif handler_result.get("type") == "function_call":
            print(f"[INFO] 检测到工具调用并已执行，准备下一轮LLM调用以获取最终回复。")
            current_messages = handler_result.get("messages")
            # 下一轮必须用 tool_choice="none" 来强制LLM回复文本
            next_llm_response_dict = get_response(current_messages, tool_choice="none")
            
            # 再次使用 handle_response 解析这轮强制无工具的响应
            # 此时的 original_messages 是包含了工具结果的消息列表
            final_handler_result = function_handler.handle_response(next_llm_response_dict, current_messages)
            
            print(f"[DEBUG] FunctionCallHandler.handle_response 结果 (after tool_choice='none'): {final_handler_result}")
            
            if final_handler_result.get("type") == "text":
                final_answer = final_handler_result.get("content", "")
                print(f"\n最终答案 (after tool_choice='none'): {final_answer}")
            elif final_handler_result.get("tool_calls"):
                print("[ERROR] LLM 在 tool_choice='none' 后仍然要求工具调用，流程异常。")
                print(f"相关工具调用请求: {final_handler_result.get('tool_calls')}")
            else:
                final_answer = final_handler_result.get("content", "未能从最终响应中提取到文本内容。")
                print(f"\n最终答案 (存在问题): {final_answer}")
            break # 结束对话
        else:
            print(f"[ERROR] 未知的 handle_response 类型: {handler_result.get('type')}")
            print(f"原始 handler_result: {handler_result}")
            break

        if turn == max_turns:
            print("\n已达到最大对话轮次。")

if __name__ == "__main__":
    call_with_messages()
