# nezha

nezha 是一个命令行 AI 助手工具，支持模型选择、对话、计划生成、配置初始化等多种智能操作，适合开发者和 AI 爱好者快速集成和使用。

## 主要特性
- 支持多种大语言模型（LLM），包括预置和用户自定义模型
- 模型可配置、可切换
- 命令行交互式体验，支持对话 (`chat`)、计划 (`plan`) 等多种命令
- 支持通过 `init` 命令初始化或重置配置
- 配置灵活，支持用户自定义模型参数和安全设置
- 内置安全层，提供不同级别的操作确认和路径限制
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
- 安全相关配置位于 `config/security_config.yaml`，详情请参考 [安全层设计与使用指南](docs/security_layer.md)

## 自动化发布流程

项目使用 GitHub Actions 实现了自动化的版本管理和发布流程，简化了发布过程。

### 自动版本管理

当代码推送到 main 分支时，系统会根据提交信息自动确定版本增量类型：

- 提交包含 "BREAKING CHANGE" 或 "major" 时，增加主版本号
- 提交包含 "feat"、"feature" 或 "minor" 时，增加次版本号
- 其他情况增加修订版本号

### 自动发布过程

工作流程会自动执行以下操作：

1. 自动更新 pyproject.toml 中的版本号
2. 提交版本更新到仓库
3. 创建并推送新版本的 git tag
4. 构建 Python 包
5. 发布到 PyPI

### 使用方法

只需正常开发和提交代码，在提交信息中使用关键词控制版本增量。当代码合并到 main 分支后，工作流程会自动执行所有步骤。

如果不希望某次提交触发发布，可以在提交信息中添加 "[skip ci]"。

## 贡献指南
欢迎提交 issue 和 PR！如需贡献代码，请遵循本项目的代码规范。

## 项目结构

### 目录结构

```
src/nezha_agent/
├── core/                 # 核心功能和基础组件
│   ├── common/          # 通用工具和功能
│   ├── context/         # 上下文引擎
│   ├── models/          # 模型接口和实现
│   ├── security/        # 安全相关功能
│   └── tools/           # 核心工具接口
├── features/            # 具体功能实现
│   ├── agent/           # 代理相关功能
│   ├── commands/        # 命令实现（chat、plan等）
│   ├── editing/         # 文件编辑功能
│   ├── git/             # Git相关功能
│   ├── shell/           # Shell相关功能
│   └── tools/           # 具体工具实现
├── integrations/        # 与外部系统的集成
│   ├── api/             # API集成
│   ├── editors/         # 编辑器集成
│   ├── external/        # 外部工具集成
│   └── shells/          # Shell集成
├── plugins/             # 插件系统
│   ├── registry/        # 插件注册管理
│   └── sdk/             # 插件开发工具包
└── ui/                  # 用户界面
    ├── cli/             # 命令行界面
    ├── display/         # 显示相关功能
    └── terminal/        # 终端相关功能
```

### 架构图

<div align="center">
  <img src="docs/nezha_architecture.svg" alt="Nezha 架构图" width="800" style="max-width: 800px;" />
</div>

架构说明：
- **Core**：核心模块，包含模型接口、安全层、上下文引擎等基础组件
- **Features**：实现具体功能，如代理、命令和各种工具
- **Integrations**：负责与外部系统和工具的交互和集成
- **Plugins**：提供插件系统，使得功能可扩展
- **UI**：用户界面模块，主要包含命令行界面和显示功能

## 联系方式
- 邮箱：137844255@qq.com
- Issues：https://github.com/echovic/nezha/issues
