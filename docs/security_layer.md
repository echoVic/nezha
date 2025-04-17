# nezha 安全层设计与使用指南

## 1. 概述

安全层 (`security.py`) 是 nezha 的核心组件之一，负责集中处理所有危险操作的用户确认逻辑，确保高风险操作（如文件写入、编辑、Shell命令执行等）在执行前得到适当的安全检查和用户确认。

安全层的主要功能包括：

- 提供不同级别的安全策略配置
- 实现用户交互确认机制
- 限制可操作的文件路径范围
- 支持禁用特定高风险工具
- 为工具系统提供安全检查接口

## 2. 核心组件

### 2.1 安全级别 (SecurityLevel)

安全层定义了四种安全级别，用于控制确认和禁用策略：

- **STRICT (严格模式)**: 最严格的安全级别，所有高风险操作都需要确认，极高风险操作（如Shell命令）可能被禁用
- **NORMAL (标准模式)**: 默认安全级别，中等及以上风险操作需要确认
- **RELAXED (宽松模式)**: 只有高风险及以上操作需要确认
- **BYPASS (跳过模式)**: 跳过大部分确认，仅极高风险操作需要确认（危险！仅用于自动化脚本）

### 2.2 工具风险级别 (ToolRiskLevel)

工具按风险级别分为五类：

- **NONE**: 无风险，如只读操作（FileRead, Ls, Glob）
- **LOW**: 低风险，如创建新文件
- **MEDIUM**: 中等风险，如修改文件（FileWrite, FileEdit）
- **HIGH**: 高风险，如删除文件
- **CRITICAL**: 极高风险，如执行Shell命令（Bash）

### 2.3 安全管理器 (SecurityManager)

安全管理器是安全层的核心类，提供以下主要方法：

- `is_tool_allowed()`: 检查工具是否被允许使用
- `is_path_allowed()`: 检查路径是否在允许的范围内
- `confirm_action()`: 请求用户确认操作

## 3. 配置方法

安全层可通过配置文件（如 `config/security_config.yaml`）或代码方式进行配置：

### 3.1 配置文件示例

```yaml
# 安全级别: strict(严格), normal(标准), relaxed(宽松), bypass(跳过确认)
security_level: normal

# 是否对所有确认自动回答是（危险！仅用于自动化脚本）
yes_to_all: false

# 允许操作的路径列表（为空表示不限制）
allowed_paths:
  - ~/Documents/projects/  # 只允许在此目录下操作

# 禁用的工具列表
disabled_tools:
  # - Bash                 # 禁用Shell命令执行
```

### 3.2 代码配置示例

```python
from nezha.security import SecurityManager, SecurityLevel

# 创建自定义安全管理器
custom_security = SecurityManager(
    security_level=SecurityLevel.STRICT,
    allowed_paths=["~/Documents/projects/"],
    disabled_tools=["Bash"],
    yes_to_all=False
)

# 替换默认安全管理器
from nezha.security import security_manager
security_manager = custom_security
```

## 4. 与工具系统集成

安全层与工具系统的集成通过以下方式实现：

1. 在工具执行前调用安全管理器的方法进行检查
2. 对高风险工具创建安全包装类

### 4.1 安全工具包装示例

```python
class SecureFileWrite(FileWrite):
    def execute(self, path, content):
        # 检查路径是否在允许范围内
        if not security_manager.is_path_allowed(path):
            return f"安全错误: 路径 '{path}' 不在允许操作的范围内"
        
        # 请求用户确认
        if not security_manager.confirm_action(
            message=f"即将写入文件 '{path}'",
            risk_level=ToolRiskLevel.MEDIUM
        ):
            return "操作已取消"
        
        # 调用原始方法执行写入
        return super().execute(path, content)
```

## 5. 最佳实践

### 5.1 安全级别选择

- 开发环境: 可使用 **RELAXED** 级别减少确认次数
- 生产环境: 建议使用 **NORMAL** 或 **STRICT** 级别
- 自动化脚本: 可使用 **BYPASS** 级别并设置 `yes_to_all=True`，但需谨慎

### 5.2 路径限制

强烈建议设置 `allowed_paths` 限制可操作的文件范围，特别是在生产环境中。

### 5.3 工具禁用

对于特别敏感的环境，可以通过 `disabled_tools` 完全禁用高风险工具，如 `Bash`。

## 6. 调试与故障排除

如果遇到安全相关的问题，可以：

1. 检查安全配置是否正确加载
2. 临时降低安全级别进行测试
3. 查看日志中的安全相关信息

## 7. 未来扩展

安全层的潜在扩展方向：

- 添加更细粒度的权限控制
- 实现操作审计日志
- 支持基于正则表达式的命令过滤
- 集成第三方安全扫描工具