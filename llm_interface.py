import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import yaml

try:
    from openai import OpenAI, OpenAIError
except ImportError:
    OpenAI = None # type: ignore
    OpenAIError = None # type: ignore


import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import yaml

# 工具描述自动注入
from tool_registry import ToolRegistry, run_tool
import json

def get_all_tool_descriptions() -> str:
    """
    自动收集所有注册工具的描述信息，拼接为 prompt 注入字符串。
    """
    registry = ToolRegistry()
    descs = []
    for tool in registry.tools.values():
        descs.append(f"工具名: {tool.name}\n用途: {getattr(tool, 'description', '')}\n参数: {getattr(tool, 'arguments', {})}\n")
    return "\n".join(descs)

def parse_llm_tool_call(llm_output: str):
    """
    尝试解析 LLM 输出的结构化工具调用意图（JSON 格式）。
    成功则自动调用工具并返回结果，否则返回 None。
    """
    try:
        data = json.loads(llm_output)
        if "tool_call" in data:
            tool_name = data["tool_call"].get("tool_name")
            args = data["tool_call"].get("args", {})
            result = run_tool(tool_name, args)
            return f"[工具 {tool_name} 调用结果]\n{result}"
    except Exception:
        pass
    return None

class LLMInterfaceBase(ABC):
    """
    LLM API 抽象基类，定义通用接口。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.api_key = self.config.get("api_key")
        self.model = self.config.get("model")
        # Use 'endpoint' instead of 'api_base' for consistency with config
        self.api_base = self.config.get("endpoint") 
        self.extra_params = self.config.get("extra_params", {})
        self.client = None # Initialize client later

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成 LLM 响应 (通常用于非聊天模型)
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
        # Extract the llm part of the config if present
        llm_config = config.get("llm", config) 
        return cls(llm_config)

# 示例：OpenAI 子类
class OpenAILLM(LLMInterfaceBase):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not OpenAI:
            raise ImportError("openai 库未安装，请运行 'pip install openai>=1.0'")
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=self.api_base # Pass base_url if provided in config
        )

    def generate(self, prompt: str, **kwargs) -> str:
        # Note: OpenAI v1.x prefers chat completions even for single prompts
        # This implementation might need adjustment based on specific use cases
        # or stick to using the chat method.
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)

    def chat(self, messages: list, **kwargs) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.config.get("max_tokens", 2048)),
                temperature=kwargs.get("temperature", self.config.get("temperature", 0.7)),
                stream=kwargs.get("stream", False),
                **self.extra_params
            )
            if kwargs.get("stream", False):
                # Handle streaming response (example: concatenate content)
                content = ""
                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                return content
            else:
                return completion.choices[0].message.content.strip()
        except OpenAIError as e:
            # TODO: Add more specific error handling
            print(f"OpenAI API error: {e}")
            raise

# 火山引擎 LLM 接口 (使用 OpenAI SDK)
class VolcEngineLLM(LLMInterfaceBase):
    DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not OpenAI:
            raise ImportError("openai 库未安装，请运行 'pip install openai>=1.0'")
        
        # 完全按照官方文档实现
        # 从环境变量获取 API Key，如果配置中也提供了，优先使用配置中的
        api_key = self.api_key or os.environ.get("ARK_API_KEY")
        if not api_key:
            raise ValueError("未在配置文件或环境变量中找到火山引擎 API Key (api_key 或 ARK_API_KEY)")
        
        # 确保 base_url 格式正确（移除末尾斜杠）
        base_url = self.api_base or self.DEFAULT_BASE_URL
        if base_url.endswith('/'):
            base_url = base_url[:-1]
            # print(f"\n注意: 移除 API 端点末尾斜杠，使用: {base_url}")
        
        # 打印调试信息
        # print(f"\n初始化火山引擎客户端:")
        # print(f"- API 端点: {base_url}")
        # print(f"- 模型 ID: {self.model}")
        # print(f"- API 密钥: {api_key[:4]}...{api_key[-4:] if api_key and len(api_key) > 8 else ''}")
        # print(f"- 环境变量 ARK_API_KEY: {'已设置' if os.environ.get('ARK_API_KEY') else '未设置'}")
        
        # 初始化客户端，不进行测试连接
        # 火山引擎不支持 /models 路径，不能使用 models.list() 进行测试
        # print("\n初始化火山引擎客户端...")
        
        try:
            # 直接初始化客户端，不进行测试连接
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )
            # print("客户端初始化成功!")
            
            # 设置环境变量，以防其他地方需要
            if not os.environ.get("ARK_API_KEY"):
                os.environ["ARK_API_KEY"] = api_key
                # print(f"已设置环境变量 ARK_API_KEY={api_key[:4]}...{api_key[-4:] if api_key and len(api_key) > 8 else ''}")
        except Exception as e:
            # print(f"\n初始化客户端时出错: {e}")
            # print(f"错误类型: {type(e).__name__}")
            pass
        # 模型 ID 是必须的
        if not self.model:
            raise ValueError("未在配置中指定火山引擎模型 (model)。")

    def generate(self, prompt: str, **kwargs) -> str:
        # 使用 chat 接口模拟 generate
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)

    def chat(self, messages: list, **kwargs) -> str:
        try:
            # print(f"\n--- 火山引擎调用信息 ---")
            # print(f"模型ID: {self.model}")
            # print(f"API端点: {self.client.base_url}")
            # print(f"API密钥: {self.api_key[:4]}...{self.api_key[-4:] if self.api_key and len(self.api_key) > 8 else ''}")
            # print(f"消息数量: {len(messages)}")
            # print(f"消息内容示例: {messages[0]['content'][:30]}..." if messages and len(messages) > 0 and 'content' in messages[0] else "无消息内容")
            
            # 参数简化，与测试脚本保持一致
            max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 500))
            temperature = kwargs.get("temperature", self.config.get("temperature", 0.7))
            
            # print(f"参数: max_tokens={max_tokens}, temperature={temperature}")
            
            # 简化请求，去除不必要的参数
            # print("\n发送纯文本请求...")
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
                # 去除其他可能引起问题的参数
            )
            
            # print("\n成功收到响应!")
            
            # 处理响应
            if hasattr(completion, "choices") and completion.choices and len(completion.choices) > 0:
                content = completion.choices[0].message.content.strip()
                # print(f"响应内容开头: {content[:50]}...")
                return content
            else:
                # print("警告: 收到意外的火山引擎响应格式。")
                # print(f"响应内容: {completion}")
                return "无法获取响应内容"
                
        except OpenAIError as e:
            # 详细的错误信息
            # print(f"\n--- 火山引擎API错误 ---")
            # print(f"错误类型: {type(e).__name__}")
            # print(f"错误信息: {str(e)}")
            
            # 尝试提取更多错误信息
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                try:
                    import json
                    error_detail = json.loads(e.response.text)
                    # print(f"错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                except:
                    # print(f"原始错误响应: {e.response.text}")
                    pass
            # 网络连接错误的特殊处理
            if "Connection error" in str(e):
                # print("\n可能的原因:")
                # print("1. 网络连接问题 - 无法连接到火山引擎API服务器")
                # print("2. API端点错误 - 配置中的endpoint可能有误")
                # print("3. 防火墙/代理问题 - 网络环境限制了对外部API的访问")
                # print("4. API服务不可用 - 火山引擎服务可能暂时不可用")
                pass
            
            return f"Error: {str(e)}"
            
        except Exception as e:
            # print(f"\n--- 火山引擎调用过程中发生意外错误 ---")
            # print(f"错误类型: {type(e).__name__}")
            # print(f"错误信息: {str(e)}")
            return f"Unexpected error: {str(e)}"



# 预留：Anthropic、其他 LLM 子类可类似扩展

class AnthropicLLM(LLMInterfaceBase):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # TODO: Add Anthropic specific initialization if needed
        # Consider using the official anthropic library

    def generate(self, prompt: str, **kwargs) -> str:
        # TODO: Implement Anthropic generate API call
        raise NotImplementedError("Anthropic generate not implemented yet")

    def chat(self, messages: list, **kwargs) -> str:
        # TODO: Implement Anthropic chat API call
        raise NotImplementedError("Anthropic chat not implemented yet")


class WenxinLLM(LLMInterfaceBase):
    """百度文心千帆 LLM 接口"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # TODO: Add Wenxin specific initialization (e.g., access token handling)
        # Consider using the official qianfan library

    def generate(self, prompt: str, **kwargs) -> str:
        # TODO: Implement Wenxin generate API call
        raise NotImplementedError("Wenxin generate not implemented yet")

    def chat(self, messages: list, **kwargs) -> str:
        # TODO: Implement Wenxin chat API call
        raise NotImplementedError("Wenxin chat not implemented yet")


class TongyiLLM(LLMInterfaceBase):
    """阿里通义千问 LLM 接口"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # TODO: Add Tongyi specific initialization
        # Consider using the official dashscope library

    def generate(self, prompt: str, **kwargs) -> str:
        # TODO: Implement Tongyi generate API call
        raise NotImplementedError("Tongyi generate not implemented yet")

    def chat(self, messages: list, **kwargs) -> str:
        # TODO: Implement Tongyi chat API call
        raise NotImplementedError("Tongyi chat not implemented yet")


class ZhipuAILLM(LLMInterfaceBase):
    """智谱AI LLM 接口"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # TODO: Add ZhipuAI specific initialization
        # Consider using the official zhipuai library

    def generate(self, prompt: str, **kwargs) -> str:
        # TODO: Implement ZhipuAI generate API call
        raise NotImplementedError("ZhipuAI generate not implemented yet")

    def chat(self, messages: list, **kwargs) -> str:
        # TODO: Implement ZhipuAI chat API call
        raise NotImplementedError("ZhipuAI chat not implemented yet")


# LLM 工厂函数，根据配置动态选择接口
def get_llm_interface(config: Dict[str, Any]) -> LLMInterfaceBase:
    # config 必须是 llm dict（包含 provider、api_key、model、endpoint）
    provider = config.get("provider", "openai").lower()

    if provider == "openai":
        return OpenAILLM(config)
    elif provider == "volcengine":
        return VolcEngineLLM(config)
    elif provider == "anthropic":
        return AnthropicLLM(config)
    elif provider == "wenxin":
        return WenxinLLM(config)
    elif provider == "tongyi":
        return TongyiLLM(config)
    elif provider == "zhipuai":
        return ZhipuAILLM(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")