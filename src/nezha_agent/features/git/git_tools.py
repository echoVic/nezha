"""
git 相关工具集合
"""
import subprocess
from .base import BaseTool
from typing import Optional

class GitStatus(BaseTool):
    name = "GitStatus"
    description = "显示当前工作区的 git 状态"
    arguments = {}
    def execute(self):
        try:
            result = subprocess.run(["git", "status"], capture_output=True, text=True, timeout=20, check=False)
            return result.stdout or result.stderr
        except Exception as e:
            return f"执行出错: {e}"

class GitLog(BaseTool):
    name = "GitLog"
    description = "显示 git 提交日志"
    arguments = {"max_count": {"type": "int", "required": False, "desc": "显示的最大提交数"}}
    def execute(self, max_count: Optional[int] = None):
        try:
            cmd = ["git", "log"]
            if max_count is not None:
                if not isinstance(max_count, int):
                    return "参数 max_count 必须为整数"
                cmd += ["-n", str(max_count)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
            return result.stdout or result.stderr
        except Exception as e:
            return f"执行出错: {e}"

class GitDiff(BaseTool):
    name = "GitDiff"
    description = "显示当前工作区的 git diff"
    arguments = {"path": {"type": "str", "required": False, "desc": "指定 diff 路径"}}
    def execute(self, path: Optional[str] = None):
        try:
            cmd = ["git", "diff"]
            if path:
                if not isinstance(path, str):
                    return "参数 path 必须为字符串"
                cmd.append(path)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
            return result.stdout or result.stderr
        except Exception as e:
            return f"执行出错: {e}"

class GitBranch(BaseTool):
    name = "GitBranch"
    description = "显示或切换 git 分支"
    arguments = {"branch": {"type": "str", "required": False, "desc": "切换到的分支名"}}
    def execute(self, branch: Optional[str] = None):
        try:
            if branch:
                if not isinstance(branch, str):
                    return "参数 branch 必须为字符串"
                cmd = ["git", "checkout", branch]
            else:
                cmd = ["git", "branch"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
            return result.stdout or result.stderr
        except Exception as e:
            return f"执行出错: {e}"

class GitPull(BaseTool):
    name = "GitPull"
    description = "执行 git pull 拉取远程更新"
    arguments = {
        "remote": {"type": "str", "required": False, "desc": "远程仓库名"},
        "branch": {"type": "str", "required": False, "desc": "分支名"}
    }
    def execute(self, remote: Optional[str] = None, branch: Optional[str] = None):
        try:
            cmd = ["git", "pull"]
            if remote:
                if not isinstance(remote, str):
                    return "参数 remote 必须为字符串"
                cmd.append(remote)
            if branch:
                if not isinstance(branch, str):
                    return "参数 branch 必须为字符串"
                cmd.append(branch)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            return result.stdout or result.stderr
        except Exception as e:
            return f"执行出错: {e}"

class GitPush(BaseTool):
    name = "GitPush"
    description = "执行 git push 推送本地提交到远程"
    arguments = {
        "remote": {"type": "str", "required": False, "desc": "远程仓库名"},
        "branch": {"type": "str", "required": False, "desc": "分支名"}
    }
    def execute(self, remote: Optional[str] = None, branch: Optional[str] = None):
        try:
            cmd = ["git", "push"]
            if remote:
                if not isinstance(remote, str):
                    return "参数 remote 必须为字符串"
                cmd.append(remote)
            if branch:
                if not isinstance(branch, str):
                    return "参数 branch 必须为字符串"
                cmd.append(branch)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            return result.stdout or result.stderr
        except Exception as e:
            return f"执行出错: {e}"

class GitAdd(BaseTool):
    name = "GitAdd"
    description = "添加文件到暂存区"
    arguments = {
        "paths": {"type": "list[str]", "required": True, "desc": "要添加的文件路径列表，至少一个"}
    }
    def execute(self, paths):
        if not paths or not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            return "参数 paths 必须为字符串列表，且不能为空"
        try:
            cmd = ["git", "add"] + paths
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
            return result.stdout or result.stderr or "添加成功"
        except Exception as e:
            return f"执行出错: {e}"

class GitCommit(BaseTool):
    name = "GitCommit"
    description = "提交暂存区到仓库"
    arguments = {
        "message": {"type": "str", "required": True, "desc": "提交信息"},
        "all": {"type": "bool", "required": False, "desc": "是否 git commit -a"}
    }
    def execute(self, message, all: Optional[bool] = False):
        if not message or not isinstance(message, str):
            return "参数 message 必须为字符串，且不能为空"
        try:
            cmd = ["git", "commit", "-m", message]
            if all:
                cmd.insert(2, "-a")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
            return result.stdout or result.stderr or "提交成功"
        except Exception as e:
            return f"执行出错: {e}"

class GitTag(BaseTool):
    name = "GitTag"
    description = "创建或列出 git 标签"
    arguments = {
        "name": {"type": "str", "required": False, "desc": "标签名，为空则列出所有标签"},
        "message": {"type": "str", "required": False, "desc": "标签说明，创建标签时可选"}
    }
    def execute(self, name: Optional[str] = None, message: Optional[str] = None):
        try:
            if name:
                if not isinstance(name, str):
                    return "参数 name 必须为字符串"
                cmd = ["git", "tag", name]
                if message:
                    if not isinstance(message, str):
                        return "参数 message 必须为字符串"
                    cmd += ["-m", message]
            else:
                cmd = ["git", "tag"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
            return result.stdout or result.stderr
        except Exception as e:
            return f"执行出错: {e}"

class GitRebase(BaseTool):
    name = "GitRebase"
    description = "变基操作（rebase）"
    arguments = {
        "branch": {"type": "str", "required": True, "desc": "要 rebase 的目标分支名"}
    }
    def execute(self, branch):
        if not branch or not isinstance(branch, str):
            return "参数 branch 必须为字符串，且不能为空"
        try:
            cmd = ["git", "rebase", branch]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            return result.stdout or result.stderr
        except Exception as e:
            return f"执行出错: {e}"
