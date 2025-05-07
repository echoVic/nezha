import os
import json
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import yaml

try:
    from openai import OpenAI, OpenAIError
    import httpx
except ImportError:
    OpenAI = None # type: ignore
    OpenAIError = None # type: ignore
    httpx = None # type: ignore

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
        
        # 从配置中获取 SSL 验证设置
        self.verify_ssl = self._parse_verify_ssl_config()
        
    def _parse_verify_ssl_config(self) -> bool:
        """从配置中解析 SSL 验证设置"""
        # 兼容 llm 嵌套和顶层 verify_ssl
        if "verify_ssl" in self.config:
            _verify_ssl_raw = self.config["verify_ssl"]
        elif "llm" in self.config and isinstance(self.config["llm"], dict) and "verify_ssl" in self.config["llm"]:
            _verify_ssl_raw = self.config["llm"]["verify_ssl"]
        else:
            return True  # 默认启用 SSL 验证
            
        # 处理不同类型的值
        if isinstance(_verify_ssl_raw, bool):
            return _verify_ssl_raw
        elif isinstance(_verify_ssl_raw, str):
            return _verify_ssl_raw.strip().lower() in ("true", "1", "yes")
        else:
            return bool(_verify_ssl_raw)
    
    def _handle_api_error(self, error: Exception, provider_name: str = "API") -> str:
        """通用 API 错误处理逻辑"""
        error_msg = str(error)
        
        # 网络连接错误处理
        if any(keyword in error_msg.lower() for keyword in ["connect", "connection", "timeout"]):
            try:
                import requests
                # 测试基本网络连接
                requests.get("https://www.baidu.com", timeout=5)
                # 如果基本网络正常，可能是API端点问题
                api_key_preview = self.api_key[:5] + "..." if self.api_key else "未设置"
                return (f"Error: Connection error. 无法连接到{provider_name}服务器({self.api_base})。请检查:\n"
                        f"1. API端点配置是否正确\n"
                        f"2. 网络设置（防火墙、代理）是否允许访问该地址\n"
                        f"3. API密钥({api_key_preview})是否正确且有效\n"
                        f"4. 模型ID({self.model})是否正确")
            except ImportError:
                return (f"Error: Connection error. 无法连接到{provider_name}服务器({self.api_base})。\n"
                       f"(提示: 安装 'requests' 库以获取更详细的网络诊断信息: pip install requests)")
            except Exception:
                # 基本网络连接也有问题
                return f"Error: Connection error. 网络连接异常，请检查您的网络连接是否正常。"
        
        # API密钥错误处理
        elif any(keyword in error_msg.lower() for keyword in ["key", "auth", "认证"]):
            api_key_preview = self.api_key[:5] + "..." if self.api_key else "未设置"
            return (f"Error: API key error. API密钥认证失败。请检查:\n"
                    f"1. API密钥({api_key_preview})是否正确\n"
                    f"2. 该密钥是否有权限访问模型({self.model})")
        
        # SSL证书错误处理
        elif any(keyword in error_msg.lower() for keyword in ["certificate", "ssl", "证书"]):
            return "Error: SSL证书验证失败。请在配置文件中添加 'verify_ssl: false' 关闭证书验证（仅用于测试环境）。"
        
        # 其他错误
        return f"Error: {error_msg}"

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

# 创建一个 OpenAI 兼容 API 的中间基类
class OpenAIBasedLLM(LLMInterfaceBase):
    """使用 OpenAI 兼容 API 的 LLM 基类"""
    DEFAULT_BASE_URL = None  # 子类应该覆盖这个值
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not OpenAI:
            raise ImportError("openai 库未安装，请运行 'pip install openai>=1.0'")
        
        # 初始化客户端
        self._init_client()
    
    def _init_client(self):
        """初始化 OpenAI 客户端"""
        # 确保 base_url 格式正确（移除末尾斜杠）
        base_url = self.api_base or self.DEFAULT_BASE_URL
        if base_url and base_url.endswith('/'):
            base_url = base_url[:-1]
            
        # 检查 API 密钥
        if not self.api_key:
            raise ValueError(f"未在配置文件中找到 API Key (api_key)")
            
        # 处理 SSL 验证
        if not self.verify_ssl and httpx:
            try:
                import urllib3
                # 禁用 SSL 证书验证的警告
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                warnings.warn("警告: SSL 证书验证已禁用，仅用于测试环境。生产环境请启用 SSL 验证。")
                
                # 创建带有 SSL 验证设置的客户端
                self.client = OpenAI(
                    base_url=base_url,
                    api_key=self.api_key,
                    http_client=httpx.Client(verify=self.verify_ssl)
                )
            except ImportError:
                # 如果没有 urllib3，仍然创建客户端但不禁用警告
                self.client = OpenAI(
                    base_url=base_url,
                    api_key=self.api_key,
                    http_client=httpx.Client(verify=self.verify_ssl)
                )
        else:
            # 正常创建客户端（启用 SSL 验证）
            self.client = OpenAI(
                base_url=base_url,
                api_key=self.api_key
            )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """使用聊天接口模拟 generate 功能"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def _process_completion_response(self, completion, stream=False):
        """处理 API 返回的 completion 对象"""
        if stream:
            # 处理流式响应
            content = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
            return content
        else:
            # 处理普通响应
            if hasattr(completion, "choices") and completion.choices and len(completion.choices) > 0:
                return completion.choices[0].message.content.strip()
            return "Error: 模型未返回有效内容"
    
    def chat(self, messages: list, **kwargs) -> str:
        """实现通用的 chat 接口"""
        try:
            # 获取参数，优先使用传入的参数，其次使用配置中的参数，最后使用默认值
            max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 2048))
            temperature = kwargs.get("temperature", self.config.get("temperature", 0.7))
            stream = kwargs.get("stream", False)
            
            # 创建请求参数
            params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
                **self.extra_params  # 添加额外参数
            }
            
            # 移除值为 None 的参数
            params = {k: v for k, v in params.items() if v is not None}
            
            # 调用 API
            completion = self.client.chat.completions.create(**params)
            
            # 处理响应
            return self._process_completion_response(completion, stream)
            
        except OpenAIError as error:
            # 使用通用错误处理
            return self._handle_api_error(error, "OpenAI")
        except Exception as error:
            # 处理其他未预期的错误
            return f"Error: Unexpected error: {str(error)}"

# OpenAI 子类
class OpenAILLM(OpenAIBasedLLM):
    """OpenAI API 接口"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

# 火山引擎 LLM 接口 (使用 OpenAI SDK)
class VolcEngineLLM(OpenAIBasedLLM):
    """火山引擎 LLM 接口，使用 OpenAI 兼容 API"""
    DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 从环境变量获取 API Key，如果配置中也提供了，优先使用配置中的
        config = config or {}
        if not config.get("api_key"):
            api_key = os.environ.get("ARK_API_KEY")
            if api_key:
                config["api_key"] = api_key
        
        # 初始化父类
        super().__init__(config)
        
        # 设置环境变量，以防其他地方需要
        if self.api_key and not os.environ.get("ARK_API_KEY"):
            try:
                os.environ["ARK_API_KEY"] = self.api_key
            except (TypeError, ValueError):
                pass
        
        # 模型 ID 是必须的
        if not self.model:
            raise ValueError("未在配置中指定火山引擎模型 (model)。")
    
    def chat(self, messages: list, **kwargs) -> str:
        # 火山引擎的默认 max_tokens 与 OpenAI 不同
        if "max_tokens" not in kwargs and "max_tokens" not in self.config:
            kwargs["max_tokens"] = 500  # 火山引擎默认值
            
        # 使用父类的 chat 方法
        try:
            return super().chat(messages, **kwargs)
        except Exception as error:
            # 处理火山引擎特有的错误
            error_msg = str(error)
            if "certificate verify failed" in error_msg.lower() or "ssl" in error_msg.lower():
                return "Error: SSL证书验证失败。请在配置文件中添加 'verify_ssl: false' 关闭证书验证（仅用于测试环境）。"
            return self._handle_api_error(error, "火山引擎")


# 预留：Anthropic、其他 LLM 子类可类似扩展

# 非 OpenAI 兼容的 LLM 接口
class AnthropicLLM(LLMInterfaceBase):
    """人类智慧公司 Anthropic 的 Claude 模型接口"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # 检查必要的配置
        if not self.api_key:
            raise ValueError("未在配置文件中找到 Anthropic API Key (api_key)")
        
        # TODO: 使用官方 anthropic 库初始化客户端
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic 库未安装，请运行 'pip install anthropic'")

    def generate(self, prompt: str, **kwargs) -> str:
        """使用聊天接口模拟 generate 功能"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)

    def chat(self, messages: list, **kwargs) -> str:
        try:
            # 获取参数
            max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 1024))
            temperature = kwargs.get("temperature", self.config.get("temperature", 0.7))
            model = self.model or "claude-3-opus-20240229"
            
            # 转换消息格式（如果需要）
            # TODO: 实现消息格式转换逻辑
            
            # 调用 API
            response = self.client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # 处理响应
            return response.content[0].text
        except ImportError as error:
            return f"Error: Anthropic 库导入错误: {error}"
        except ValueError as error:
            return f"Error: 参数错误: {error}"
        except ConnectionError as error:
            return f"Error: 连接错误: {error}"
        except AttributeError as error:
            return f"Error: 属性错误: {error}"
        except Exception as error:
            # 使用通用错误处理，仅在其他异常都不匹配时使用
            return self._handle_api_error(error, "Anthropic")


# 中国本土 LLM 接口类的基类
class ChineseLLMBase(LLMInterfaceBase):
    """中国本土 LLM 接口的基类，包含共享功能"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # 检查必要的配置
        if not self.api_key:
            raise ValueError(f"未在配置文件中找到 {self.__class__.__name__} API Key (api_key)")
        
        # 初始化客户端
        self._init_client()
    
    def _init_client(self):
        """初始化客户端，子类应覆盖此方法"""
        # 这是一个抽象方法，由子类实现
    
    def generate(self, prompt: str, **kwargs) -> str:
        """使用聊天接口模拟 generate 功能"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)


class WenxinLLM(ChineseLLMBase):
    """百度文心千帆 LLM 接口"""
    
    def _init_client(self):
        """初始化百度文心千帆客户端"""
        try:
            import qianfan
            self.client = qianfan.Qianfan(ak=self.api_key, sk=self.config.get("secret_key"))
        except ImportError:
            raise ImportError("qianfan 库未安装，请运行 'pip install qianfan'")

    def chat(self, messages: list, **kwargs) -> str:
        try:
            # 获取参数
            model = self.model or "ERNIE-Bot-4"
            
            # 调用 API
            response = self.client.chat(model=model, messages=messages)
            
            # 处理响应
            return response.result
        except ImportError as error:
            return f"Error: 百度文心库导入错误: {error}"
        except ValueError as error:
            return f"Error: 参数错误: {error}"
        except ConnectionError as error:
            return f"Error: 连接错误: {error}"
        except AttributeError as error:
            return f"Error: 属性错误: {error}"
        except Exception as error:
            # 仅在其他异常都不匹配时使用通用错误处理
            return self._handle_api_error(error, "百度文心")


class TongyiLLM(ChineseLLMBase):
    """阿里通义千问 LLM 接口"""
    
    def _init_client(self):
        """初始化阿里通义千问客户端"""
        try:
            import dashscope
            dashscope.api_key = self.api_key
            self.client = dashscope
        except ImportError:
            raise ImportError("dashscope 库未安装，请运行 'pip install dashscope'")

    def chat(self, messages: list, **kwargs) -> str:
        try:
            # 获取参数
            model = self.model or "qwen-max"
            
            # 调用 API
            from dashscope import Generation
            response = Generation.call(
                model=model,
                messages=messages,
                result_format='message',
            )
            
            # 处理响应
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                return f"Error: API 调用失败 (状态码: {response.status_code})"
        except ImportError as error:
            return f"Error: 阿里通义库导入错误: {error}"
        except ValueError as error:
            return f"Error: 参数错误: {error}"
        except ConnectionError as error:
            return f"Error: 连接错误: {error}"
        except AttributeError as error:
            return f"Error: 属性错误: {error}"
        except Exception as error:
            # 仅在其他异常都不匹配时使用通用错误处理
            return self._handle_api_error(error, "阿里通义")


class ZhipuAILLM(ChineseLLMBase):
    """智谱AI LLM 接口"""
    
    def _init_client(self):
        """初始化智谱AI客户端"""
        try:
            import zhipuai
            self.client = zhipuai.ZhipuAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("zhipuai 库未安装，请运行 'pip install zhipuai'")

    def chat(self, messages: list, **kwargs) -> str:
        try:
            # 获取参数
            model = self.model or "glm-4"
            temperature = kwargs.get("temperature", self.config.get("temperature", 0.7))
            
            # 调用 API
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            
            # 处理响应
            return response.choices[0].message.content
        except ImportError as error:
            return f"Error: 智谱AI库导入错误: {error}"
        except ValueError as error:
            return f"Error: 参数错误: {error}"
        except ConnectionError as error:
            return f"Error: 连接错误: {error}"
        except AttributeError as error:
            return f"Error: 属性错误: {error}"
        except Exception as error:
            # 仅在其他异常都不匹配时使用通用错误处理
            return self._handle_api_error(error, "智谱AI")


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