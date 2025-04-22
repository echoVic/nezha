"""
Agent 核心逻辑类
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from nezha_agent.context_engine import \
    ContextEngine  # Assuming ContextEngine provides context format
from nezha_agent.llm_interface import LLMInterfaceBase, get_llm_interface
from nezha_agent.security import SecurityManager

# 导入获取当前模型的函数
try:
    from nezha_agent.cli import get_current_model
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
                from nezha_agent.cli import PREDEFINED_MODELS
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

        # 初始化 LLM 接口
        # 只保留必要的警告信息
        if not llm_config.get("verify_ssl", True):
            print("\n警告: SSL 证书验证已禁用，仅用于测试环境。生产环境请启用 SSL 验证。")
        
        self.llm_interface = get_llm_interface(llm_config)

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

    def plan_chat(self, history: list, verbose: bool = False):
        """Handles the interactive planning chat loop."""
        if verbose:
            print(f"\n--- Sending History to LLM ({self.config.get('llm', {}).get('provider')}) ---")
            for msg in history:
                print(f"[{msg['role']}]: {msg['content']}")
            print("---------------------------------------")

        try:
            # Directly use the history provided by PlanCommand
            response = self.llm_interface.chat(history)
            # if verbose:
            #     print("\n--- LLM Response ---")
            #     print(response)
            #     print("--------------------")
            return response
        except Exception as e:
            print(f"Error during LLM call in plan_chat: {e}")
            return f"Error during planning chat: {e}"

    def run(self, prompt: str, context: Optional[Dict[str, Any]] = None, verbose: bool = False):
        """执行用户指令"""
        # TODO: Implement the core agent loop:
        # 1. Construct the full prompt including context, system prompt, etc.
        # 2. Call the LLM using self.llm_interface.chat() or generate()
        # 3. Parse the LLM response (e.g., identify tool calls)
        # 4. Execute tools securely using self.security_manager
        # 5. Format and return the final result

        # Placeholder implementation:
        full_prompt = f"Context:\n{context}\n\nUser Prompt: {prompt}"
        if verbose:
            print(f"\n--- Sending to LLM ({self.config.get('llm', {}).get('provider')}) ---")
            print(full_prompt)
            print("---------------------------------------")

        try:
            # Assuming chat interface is preferred
            # Construct messages list based on LLM provider requirements
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant." }, # TODO: Load system prompt from config/prompts
                {"role": "user", "content": full_prompt}
            ]
            response = self.llm_interface.chat(messages)
            # if verbose:
            #     print("\n--- LLM Response ---")
            #     print(response)
            #     print("--------------------")
            return response
        except Exception as e:
            # TODO: More specific error handling
            # print(f"Error during LLM call: {e}")
            return f"Error executing prompt: {e}"