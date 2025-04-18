# nezha

nezha 是一个命令行 AI 助手工具，支持模型选择、对话、计划生成等多种智能操作，适合开发者和 AI 爱好者快速集成和使用。

## 主要特性
- 支持多种火山引擎大模型，模型可配置、可切换
- 命令行交互式体验，支持对话、计划等多种命令
- 配置灵活，支持用户自定义模型参数
- 适合二次开发和集成

## 安装方法

### 通过 pip 安装（推荐）
```bash
pip install nezha-agent
```

### 源码安装
```bash
git clone https://github.com/your-org/nezha.git
cd nezha
pip install .
```

## 快速开始

### 查看可用模型
```bash
nezha models
```

### 选择模型
```bash
nezha models select <模型名>
```

### 启动对话
```bash
nezha chat
```

### 生成计划
```bash
nezha plan "帮我写一个发邮件的脚本"
```

## 配置说明
- 默认配置文件位于 `~/.nezha/config.yaml`
- 可通过命令行参数或环境变量覆盖默认配置
- 支持自定义模型列表及参数

## 贡献指南
欢迎提交 issue 和 PR！如需贡献代码，请遵循本项目的代码规范。

## 联系方式
- 邮箱：your.email@example.com
- Issues：https://github.com/your-org/nezha/issues
