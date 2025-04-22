import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import json

import yaml

try:
    from openai import OpenAI, OpenAIError
except ImportError:
    OpenAI = None # type: ignore
    OpenAIError = None # type: ignore

# 工具描述自动注入
from .tool_registry import ToolRegistry, run_tool

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
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        # 处理特定的异常类型
        if isinstance(error, json.JSONDecodeError):
            # JSON 解析错误
            pass
        elif isinstance(error, (KeyError, TypeError)):
            # 数据结构或类型错误
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
            # 处理 OpenAI API 错误
            error_msg = str(e)
            if "connect" in error_msg.lower() or "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                import requests
                # 尝试检测更具体的网络问题
                try:
                    # 测试基本网络连接
                    requests.get("https://www.baidu.com", timeout=5)
                    # 如果基本网络正常，可能是API端点问题
                    return (f"Error: Connection error. 无法连接到API服务器({self.api_base})。请检查:\n"
                            f"1. API端点配置是否正确\n"
                            f"2. 网络是否可以访问该API服务器\n"
                            f"3. API密钥是否正确配置")
                except requests.exceptions.RequestException:
                    # 基本网络连接也有问题
                    return "Error: Connection error. 网络连接异常，请检查您的网络连接是否正常。"
            elif "key" in error_msg.lower() or "auth" in error_msg.lower():
                api_key_preview = self.api_key[:5] + "..." if self.api_key else "未设置"
                return (f"Error: API key error. API密钥认证失败。请检查:\n"
                        f"1. API密钥({api_key_preview})是否正确\n"
                        f"2. 该密钥是否有权限访问模型({self.model})")
            else:
                return f"Error: {error_msg}"
        except Exception as e:
            # 处理其他未预期的错误
            return f"Error: Unexpected error: {str(e)}"

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
        
        # 禁用 SSL 证书验证（仅用于测试）
        self.verify_ssl = self.config.get("verify_ssl", True)
        
        # 初始化客户端
        try:
            import urllib3
            if not self.verify_ssl:
                # 禁用 SSL 证书验证的警告
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                print("\n警告: SSL 证书验证已禁用，仅用于测试环境。生产环境请启用 SSL 验证。")
            
            # 创建 OpenAI 客户端
            import httpx
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                # 添加自定义 httpx 客户端以支持禁用 SSL 验证
                http_client=httpx.Client(verify=self.verify_ssl)
            )
            
            # 设置环境变量，以防其他地方需要
            if not os.environ.get("ARK_API_KEY"):
                try:
                    os.environ["ARK_API_KEY"] = api_key
                except (TypeError, ValueError):
                    pass
        except Exception as e:
            print(f"\n初始化客户端失败: {e}")
        
        # 模型 ID 是必须的
        if not self.model:
            raise ValueError("未在配置中指定火山引擎模型 (model)。")

    def generate(self, prompt: str, **kwargs) -> str:
        # 使用 chat 接口模拟 generate
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)

    def chat(self, messages: list, **kwargs) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.config.get("max_tokens", 500)),
                temperature=kwargs.get("temperature", self.config.get("temperature", 0.7))
            )
            
            # 处理响应
            if hasattr(completion, "choices") and completion.choices and len(completion.choices) > 0:
                content = completion.choices[0].message.content.strip()
                return content
            else:
                return "Error: 模型未返回有效内容"
            if "Connection error" in str(error):
                # print("\n可能的原因:")
                # print("1. 网络连接问题 - 无法连接到火山引擎API服务器")
                # print("2. API端点错误 - 配置中的endpoint可能有误")
                # print("3. 防火墙/代理问题 - 网络环境限制了对外部API的访问")
                # print("4. API服务不可用 - 火山引擎服务可能暂时不可用")
                pass
            
            return f"Error: {str(error)}"
            
        except (ValueError, TypeError, ConnectionError) as error:
            # 处理可预期的错误类型
            return f"Error: {str(error)}"
        except Exception as error:
            # 只在需要时输出特殊错误提示
            if "certificate verify failed" in str(error).lower() or "ssl" in str(error).lower():
                return "Error: SSL证书验证失败。请在配置文件中添加 'verify_ssl: false' 关闭证书验证（仅用于测试环境）。"
            
            return f"Error: {str(error)}"


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