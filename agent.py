"""
Agent 核心逻辑类
"""
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from context_engine import \
    ContextEngine  # Assuming ContextEngine provides context format
from llm_interface import LLMInterfaceBase, get_llm_interface
from security import SecurityManager

DEFAULT_CONFIG_PATH = Path("config/config.yaml")

class NezhaAgent:
    def __init__(self, security_manager: SecurityManager, config_file: Optional[Path] = None):
        self.security_manager = security_manager
        self.config_path = config_file or DEFAULT_CONFIG_PATH
        self.config = self._load_config()
        llm_config = self.config.get("llm", {})
        self.llm_interface: LLMInterfaceBase = get_llm_interface(llm_config)
        # TODO: Initialize other components like tool registry based on config

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                # TODO: Add proper logging/error handling
                print(f"Error loading config file {self.config_path}: {e}")
                return {}
        else:
            # TODO: Handle missing config file scenario (e.g., prompt init)
            print(f"Warning: Config file not found at {self.config_path}. Using default settings.")
            return {}

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
            if verbose:
                print("\n--- LLM Response ---")
                print(response)
                print("--------------------")
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
            if verbose:
                print("\n--- LLM Response ---")
                print(response)
                print("--------------------")
            return response
        except Exception as e:
            # TODO: More specific error handling
            print(f"Error during LLM call: {e}")
            return f"Error executing prompt: {e}"