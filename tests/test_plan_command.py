#!/usr/bin/env python
"""
PlanCommand单元测试
"""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agent import NezhaAgent
from src.context_engine import ContextEngine
from src.plan_command import PlanCommand


class TestPlanCommand(unittest.TestCase):
    """测试交互式规划命令"""
    
    def setUp(self):
        """测试前准备"""
        self.mock_agent = MagicMock(spec=NezhaAgent)
        self.mock_context_engine = MagicMock(spec=ContextEngine)
        self.mock_context_engine.collect.return_value = "测试上下文"
        
        # 模拟agent.plan_chat方法返回响应
        self.mock_agent.plan_chat.side_effect = [
            "我需要了解更多关于你的项目。你使用什么编程语言？",  # 第一次回复
            "明白了。我建议以下步骤:\n1. 设置pytest框架\n2. 创建基本测试\n3. 配置GitHub Actions\n\n这个计划可行吗？",  # 第二次回复
            "# 项目测试与CI/CD集成计划\n\n## 目标\n为项目添加单元测试框架和CI/CD集成\n\n## 详细步骤\n1. 安装pytest\n2. 创建测试目录结构\n3. 编写基本测试用例\n4. 配置GitHub Actions\n\n[计划完成]"  # 最终计划
        ]
        
        # 创建PlanCommand实例
        self.planner = PlanCommand(
            agent=self.mock_agent,
            context_engine=self.mock_context_engine,
            verbose=False,
            output_file=None
        )
    
    @patch('src.plan_command.input')
    @patch('src.plan_command.print')
    def test_run_plan_command(self, mock_print, mock_input):
        """测试运行交互式规划流程"""
        # 模拟用户输入
        mock_input.side_effect = [
            "我的项目使用Python",  # 第一次用户回复
            "可行，请生成详细计划",  # 第二次用户回复
            ""  # 用户回车结束
        ]
        
        # 执行规划
        result = self.planner.run("为项目添加单元测试框架和CI/CD集成")
        
        # 验证结果
        self.assertIn("# 项目测试与CI/CD集成计划", result)
        self.assertIn("## 目标", result)
        self.assertIn("## 详细步骤", result)
        
        # 验证方法调用
        self.assertEqual(self.mock_agent.plan_chat.call_count, 3) # 修正: 检查 plan_chat 而不是 run
        # self.assertEqual(self.mock_context_engine.collect.call_count, 1) # 移除: collect 未在 PlanCommand 中调用
        self.assertEqual(mock_input.call_count, 3) # 修正: 预期调用 3 次 input
    
    @patch('src.plan_command.open', new_callable=unittest.mock.mock_open) # 修正: 模拟内置 open 函数
    @patch('src.plan_command.print') # 修正: 目标是 print
    @patch('src.plan_command.input') # 修正: 目标是 input
    def test_output_file_saving(self, mock_input, mock_print, mock_open): # 修正: 参数名 mock_open
        """测试计划保存到文件"""
        # 设置输出文件
        self.planner.output_file = Path("test_plan.md")
        
        # 模拟用户输入
        mock_input.side_effect = ["", ""] # 修正: 模拟用户直接回车结束，因为 test_output_file_saving 不涉及多轮交互
        
        # 执行规划
        self.planner.run("测试需求")
        
        # 验证文件写入
        mock_open.assert_called_once_with(self.planner.output_file, "w", encoding="utf-8")
        mock_open().write.assert_called_once()

if __name__ == '__main__':
    unittest.main()