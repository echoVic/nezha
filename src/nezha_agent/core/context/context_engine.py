"""上下文收集与管理

该模块负责收集和管理传递给LLM的上下文信息，包括目录结构、Git信息、指定文件内容和代码风格等。
实现token预算管理，当上下文过长时采用优先级策略选择信息。
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

class ContextEngine:
    """上下文引擎类，负责收集和管理传递给LLM的上下文信息"""
    
    def __init__(self, max_tokens: int = 4000, working_dir: Optional[str] = None):
        """初始化上下文引擎
        
        Args:
            max_tokens: 上下文最大token数量
            working_dir: 工作目录，默认为当前目录
        """
        self.max_tokens = max_tokens
        self.working_dir = working_dir or os.getcwd()
        self.context_data = {}
        self.priority_levels = {
            "user_specified_files": 1,  # 用户明确指定的文件
            "git_changes": 2,          # Git变更的文件
            "key_files": 3,            # 项目关键文件
            "directory_structure": 4,   # 目录结构
            "code_style": 5,           # 代码风格
            "git_info": 6,             # Git信息
        }
    
    def collect(self, user_files: List[str] = None) -> Dict[str, Any]:
        """收集上下文信息
        
        Args:
            user_files: 用户指定的文件列表
            
        Returns:
            收集到的上下文信息字典
        """
        # 清空之前的上下文数据
        self.context_data = {}
        
        # 收集当前工作目录
        self.context_data["working_directory"] = self.working_dir
        
        # 收集目录结构
        self.collect_directory_structure()
        
        # 收集Git信息（如果可用）
        self.collect_git_info()
        
        # 收集用户指定的文件
        if user_files:
            self.collect_specified_files(user_files)
        
        # 收集项目关键文件
        self.collect_key_files()
        
        # 收集代码风格信息
        self.collect_code_style()
        
        # 应用token预算管理
        self.apply_token_budget()
        
        return self.context_data
    
    def collect_directory_structure(self, max_depth: int = 2):
        """收集目录结构
        
        Args:
            max_depth: 最大目录深度
        """
        try:
            # 尝试使用tree命令（如果可用）
            result = subprocess.run(
                ["tree", "-L", str(max_depth), "-d", self.working_dir],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                self.context_data["directory_structure"] = result.stdout
                return
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # 如果tree命令不可用，使用os.walk
        structure = []
        root_path = Path(self.working_dir)
        structure.append(str(root_path))
        
        for current_depth in range(1, max_depth + 1):
            for root, dirs, files in os.walk(self.working_dir):
                path = Path(root)
                rel_path = path.relative_to(self.working_dir)
                depth = len(rel_path.parts)
                
                if depth == current_depth - 1:
                    for d in sorted(dirs):
                        indent = "  " * depth
                        structure.append(f"{indent}├── {d}/")
        
        self.context_data["directory_structure"] = "\n".join(structure)
    
    def collect_git_info(self):
        """收集Git信息"""
        if not GIT_AVAILABLE:
            return
        
        try:
            repo = git.Repo(self.working_dir)
            git_info = {}
            
            # 获取工作区状态
            if not repo.bare:
                git_info["status"] = self._get_git_status(repo)
                
                # 获取最近的提交
                git_info["recent_commits"] = self._get_recent_commits(repo)
                
                # 获取变更的文件
                git_info["changed_files"] = self._get_changed_files(repo)
            
            self.context_data["git_info"] = git_info
        except (git.InvalidGitRepositoryError, git.NoSuchPathError):
            # 不是Git仓库或路径不存在
            pass
    
    def _get_git_status(self, repo):
        """获取Git状态"""
        return repo.git.status(porcelain=True)
    
    def _get_recent_commits(self, repo, count=5):
        """获取最近的提交"""
        commits = []
        for commit in repo.iter_commits(max_count=count):
            commits.append({
                "hash": commit.hexsha[:7],
                "message": commit.message.strip(),
                "author": f"{commit.author.name} <{commit.author.email}>",
                "date": commit.committed_datetime.isoformat(),
            })
        return commits
    
    def _get_changed_files(self, repo):
        """获取变更的文件"""
        changed_files = []
        for item in repo.index.diff(None):
            changed_files.append(item.a_path)
        return changed_files
    
    def collect_specified_files(self, file_paths: List[str]):
        """收集用户指定的文件内容
        
        Args:
            file_paths: 文件路径列表
        """
        specified_files = {}
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    specified_files[file_path] = content
            except (IOError, UnicodeDecodeError) as e:
                specified_files[file_path] = f"Error reading file: {str(e)}"
        
        self.context_data["user_specified_files"] = specified_files
    
    def collect_key_files(self):
        """收集项目关键文件"""
        key_files = [
            "README.md",
            "pyproject.toml",
            "setup.py",
            "requirements.txt",
            ".gitignore"
        ]
        
        found_key_files = {}
        
        for file_name in key_files:
            file_path = os.path.join(self.working_dir, file_name)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        found_key_files[file_name] = content
                except (IOError, UnicodeDecodeError):
                    pass
        
        self.context_data["key_files"] = found_key_files
    
    def collect_code_style(self):
        """收集代码风格信息"""
        style_files = [
            ".flake8",
            ".pylintrc",
            "pyproject.toml",  # 可能包含black、isort等配置
            ".editorconfig",
            ".style.yapf",
        ]
        
        style_info = {}
        
        for file_name in style_files:
            file_path = os.path.join(self.working_dir, file_name)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        style_info[file_name] = content
                except (IOError, UnicodeDecodeError):
                    pass
        
        self.context_data["code_style"] = style_info
    
    def apply_token_budget(self):
        """应用token预算管理，当上下文过长时采用优先级策略选择信息"""
        # 简单估算当前token数量（粗略估计：1个token约等于4个字符）
        current_tokens = self._estimate_tokens(self.context_data)
        
        if current_tokens <= self.max_tokens:
            return
        
        # 按优先级裁剪内容
        for category, priority in sorted(self.priority_levels.items(), key=lambda x: x[1], reverse=True):
            if category not in self.context_data:
                continue
                
            # 对于最高优先级的内容（用户指定的文件），尝试保留但可能需要截断
            if priority == 1 and isinstance(self.context_data[category], dict):
                for file_path, content in self.context_data[category].items():
                    if len(content) > 1000:  # 简单截断策略
                        self.context_data[category][file_path] = content[:500] + "\n...\n" + content[-500:]
            
            # 对于低优先级内容，如果仍然超出预算，则移除
            elif priority > 3:  # 目录结构及更低优先级
                current_tokens = self._estimate_tokens(self.context_data)
                if current_tokens > self.max_tokens:
                    self.context_data.pop(category, None)
    
    def _estimate_tokens(self, data) -> int:
        """粗略估计数据的token数量
        
        Args:
            data: 要估计的数据
            
        Returns:
            估计的token数量
        """
        if isinstance(data, str):
            return len(data) // 4
        elif isinstance(data, dict):
            return sum(self._estimate_tokens(k) + self._estimate_tokens(v) for k, v in data.items())
        elif isinstance(data, list):
            return sum(self._estimate_tokens(item) for item in data)
        else:
            return len(str(data)) // 4
    
    def format_context(self) -> str:
        """格式化上下文信息为字符串
        
        Returns:
            格式化后的上下文字符串
        """
        sections = []
        
        # 添加工作目录
        sections.append(f"当前工作目录: {self.context_data.get('working_directory', '未知')}\n")
        
        # 添加目录结构
        if "directory_structure" in self.context_data:
            sections.append("目录结构:\n```\n" + self.context_data["directory_structure"] + "\n```\n")
        
        # 添加Git信息
        if "git_info" in self.context_data:
            git_info = self.context_data["git_info"]
            git_section = ["Git信息:"]
            
            if "status" in git_info and git_info["status"]:
                git_section.append("状态:\n```\n" + git_info["status"] + "\n```\n")
            
            if "changed_files" in git_info and git_info["changed_files"]:
                git_section.append("变更文件:\n- " + "\n- ".join(git_info["changed_files"]) + "\n")
            
            if "recent_commits" in git_info and git_info["recent_commits"]:
                commits = git_info["recent_commits"]
                git_section.append("最近提交:")
                for commit in commits:
                    git_section.append(f"- {commit['hash']} - {commit['message']} ({commit['author']})")
            
            sections.append("\n".join(git_section) + "\n")
        
        # 添加用户指定的文件
        if "user_specified_files" in self.context_data and self.context_data["user_specified_files"]:
            files = self.context_data["user_specified_files"]
            sections.append("用户指定的文件:")
            for file_path, content in files.items():
                sections.append(f"文件: {file_path}\n```\n{content}\n```\n")
        
        # 添加关键文件
        if "key_files" in self.context_data and self.context_data["key_files"]:
            files = self.context_data["key_files"]
            sections.append("项目关键文件:")
            for file_name, content in files.items():
                sections.append(f"文件: {file_name}\n```\n{content}\n```\n")
        
        # 添加代码风格信息
        if "code_style" in self.context_data and self.context_data["code_style"]:
            sections.append("代码风格信息:")
            for file_name, content in self.context_data["code_style"].items():
                sections.append(f"配置文件: {file_name}\n```\n{content}\n```\n")
        
        return "\n".join(sections)