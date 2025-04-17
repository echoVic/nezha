import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import yaml


class LLMInterfaceBase(ABC):
    """
    LLM API 抽象基类，定义通用接口。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.api_key = self.config.get("api_key")
        self.model = self.config.get("model")
        self.api_base = self.config.get("api_base")
        self.extra_params = self.config.get("extra_params", {})

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成 LLM 响应
        """
        pass

    @abstractmethod
    def chat(self, messages: list, **kwargs) -> str:
        """
        支持多轮对话的接口
        """
        pass

    @classmethod
    def from_config_file(cls, config_path: str):
        """
        从 YAML/TOML 配置文件加载参数
        """
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        else:
            raise NotImplementedError("只支持 YAML 配置文件")
        return cls(config)

# 示例：OpenAI 子类（可扩展更多模型）
class OpenAILLM(LLMInterfaceBase):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        try:
            import openai
            self.openai = openai
        except ImportError:
            self.openai = None

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.openai:
            raise ImportError("openai 库未安装")
        response = self.openai.Completion.create(
            api_key=self.api_key,
            model=self.model,
            prompt=prompt,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            **self.extra_params
        )
        return response["choices"][0]["text"].strip()

    def chat(self, messages: list, **kwargs) -> str:
        if not self.openai:
            raise ImportError("openai 库未安装")
        response = self.openai.ChatCompletion.create(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            **self.extra_params
        )
        return response["choices"][0]["message"]["content"].strip()

# 预留：Anthropic、其他 LLM 子类可类似扩展