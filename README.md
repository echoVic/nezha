# nezha

nezha 是一个命令行 AI 助手工具，支持模型选择、对话、计划生成、配置初始化等多种智能操作，适合开发者和 AI 爱好者快速集成和使用。

## 主要特性
- 支持多种大语言模型（LLM），包括预置和用户自定义模型
- 模型可配置、可切换
- 命令行交互式体验，支持对话 (`chat`)、计划 (`plan`) 等多种命令
- 支持通过 `init` 命令初始化或重置配置
- 配置灵活，支持用户自定义模型参数和安全设置
- 适合二次开发和集成

## 安装方法

### 通过 pip 安装（推荐）
```bash
pip install nezha-agent
```

### 源码安装
```bash
git clone https://github.com/echovic/nezha.git
cd nezha
pip install .
```

## 快速开始

### 初始化配置 (可选)
```bash
nezha init
```

### 查看和管理模型
```bash
# 查看所有可用模型
nezha models list

# 交互式选择并设置默认模型
nezha models select

# 添加新模型配置
nezha models add
```

### 启动对话
```bash
nezha chat
```

### 生成计划
```bash
nezha plan "帮我写一个发送邮件的 Python 脚本"
```

## 配置说明
- 默认配置文件位于 `~/.config/nezha/config.yaml` (路径可能因操作系统而异，请参考 `platformdirs` 文档)
- 可通过 `nezha init` 命令生成或重置配置文件
- 支持自定义模型列表及参数
- 安全相关配置位于 `security_config.yaml`

## 贡献指南
欢迎提交 issue 和 PR！如需贡献代码，请遵循本项目的代码规范。

## 联系方式
- 邮箱：137844255@qq.com
- Issues：https://github.com/echovic/nezha/issues
