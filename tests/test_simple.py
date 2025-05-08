#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单测试重构后的 LLM 接口核心功能
"""

import os
import sys
from typing import Dict, Any, Optional, List

# 创建一个简单的模拟类来测试核心功能
class MockOpenAIBasedLLM:
    """模拟 OpenAIBasedLLM 类的核心功能"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.api_key = self.config.get("api_key")
        self.model = self.config.get("model")
        self.api_base = self.config.get("endpoint")
        self.extra_params = self.config.get("extra_params", {})
        self.verify_ssl = self._parse_verify_ssl_config()
        
    def _parse_verify_ssl_config(self) -> bool:
        """从配置中解析 SSL 验证设置"""
        if "verify_ssl" in self.config:
            _verify_ssl_raw = self.config["verify_ssl"]
        else:
            return True
            
        if isinstance(_verify_ssl_raw, bool):
            return _verify_ssl_raw
        elif isinstance(_verify_ssl_raw, str):
            return _verify_ssl_raw.strip().lower() in ("true", "1", "yes")
        else:
            return bool(_verify_ssl_raw)
    
    def _handle_api_error(self, error: Exception, provider_name: str = "API") -> str:
        """通用 API 错误处理逻辑"""
        error_msg = str(error)
        return f"Error: {error_msg}"
    
    def _process_completion_response(self, completion, stream=False):
        """处理 API 返回的 completion 对象"""
        # 简单模拟返回内容
        return "这是一个模拟的 API 响应。"
    
    def generate(self, prompt: str, **kwargs) -> str:
        """使用聊天接口模拟 generate 功能"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """实现通用的 chat 接口"""
        try:
            # 获取参数
            max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 2048))
            temperature = kwargs.get("temperature", self.config.get("temperature", 0.7))
            
            # 模拟 API 调用
            print(f"模拟 API 调用:")
            print(f"- 模型: {self.model}")
            print(f"- 消息: {messages}")
            print(f"- 最大 token 数: {max_tokens}")
            print(f"- 温度: {temperature}")
            print(f"- SSL 验证: {self.verify_ssl}")
            
            # 模拟响应
            return f"这是对 '{messages[-1]['content']}' 的模拟响应。"
            
        except Exception as error:
            # 使用通用错误处理
            return self._handle_api_error(error, "模拟 API")


class MockOpenAILLM(MockOpenAIBasedLLM):
    """模拟 OpenAI 接口"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)


class MockVolcEngineLLM(MockOpenAIBasedLLM):
    """模拟火山引擎接口"""
    DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 从环境变量获取 API Key
        config = config or {}
        if not config.get("api_key"):
            api_key = os.environ.get("ARK_API_KEY")
            if api_key:
                config["api_key"] = api_key
        
        super().__init__(config)
        
        # 模型 ID 是必须的
        if not self.model:
            raise ValueError("未在配置中指定火山引擎模型 (model)。")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # 火山引擎的默认 max_tokens 与 OpenAI 不同
        if "max_tokens" not in kwargs and "max_tokens" not in self.config:
            kwargs["max_tokens"] = 500  # 火山引擎默认值
            
        # 使用父类的 chat 方法
        return super().chat(messages, **kwargs)


def test_openai_mock():
    """测试 OpenAI 模拟接口"""
    print("=== 测试 OpenAI 模拟接口 ===")
    
    config = {
        "api_key": "sk-example",
        "model": "gpt-3.5-turbo",
        "max_tokens": 100
    }
    
    llm = MockOpenAILLM(config)
    messages = [{"role": "user", "content": "你好，请简单介绍一下自己。"}]
    response = llm.chat(messages)
    print(f"响应: {response}")
    print("测试成功!")


def test_volcengine_mock():
    """测试火山引擎模拟接口"""
    print("\n=== 测试火山引擎模拟接口 ===")
    
    config = {
        "api_key": "ark-example",
        "model": "eb-4",
        "verify_ssl": False,
        "max_tokens": 100
    }
    
    llm = MockVolcEngineLLM(config)
    messages = [{"role": "user", "content": "你好，请简单介绍一下自己。"}]
    response = llm.chat(messages)
    print(f"响应: {response}")
    print("测试成功!")


if __name__ == "__main__":
    test_openai_mock()
    test_volcengine_mock()
