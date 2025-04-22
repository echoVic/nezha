"""
Agent工具调用与对话流闭环实现
"""
from nezha_agent.tool_registry import run_tool
from typing import Any, Dict, Optional


def parse_tool_call(llm_response: str) -> Optional[Dict[str, Any]]:
    """
    简单解析LLM输出中的工具调用意图。
    约定格式: 
    工具调用: <tool=ToolName> <参数1=值1> <参数2=值2>
    例如: 工具调用: <tool=FileRead> <path=README.md>
    """
    import re
    pattern = r"工具调用:\s*<tool=(\w+)>((?:\s*<\w+=.*?>)*)"
    match = re.search(pattern, llm_response)
    if not match:
        return None
    tool_name = match.group(1)
    args_str = match.group(2)
    arg_pattern = r"<(\w+)=(.*?)>"
    args = dict(re.findall(arg_pattern, args_str))
    return {"tool_name": tool_name, "args": args}


def agent_with_tool_loop(agent, messages, verbose=False):
    """
    LLM回复如包含工具调用指令，则自动调用工具并追加结果到对话流。
    """
    history = messages[:]
    while True:
        response = agent.llm_interface.chat(history)
        if verbose:
            print("\n[LLM回复]:", response)
        tool_call = parse_tool_call(response)
        if tool_call:
            tool_result = run_tool(tool_call["tool_name"], tool_call["args"])
            tool_msg = f"[工具 {tool_call['tool_name']} 结果]:\n{tool_result}"
            history.append({"role": "function", "content": tool_msg})
            if verbose:
                print(f"\n[自动调用工具: {tool_call['tool_name']}]\n{tool_result}")
            # 继续循环，将工具结果反馈给LLM
            continue
        else:
            return response

# 用法示例：
# result = agent_with_tool_loop(agent, messages, verbose=True)
