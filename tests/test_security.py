#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试security模块是否正常工作
"""

from security import SecurityLevel, ToolRiskLevel, security_manager

print('导入成功！安全模块正常工作')
print(f'当前安全级别: {security_manager.security_level}')

# 测试风险级别
print(f'低风险级别: {ToolRiskLevel.LOW}')
print(f'中风险级别: {ToolRiskLevel.MEDIUM}')
print(f'高风险级别: {ToolRiskLevel.HIGH}')
print(f'极高风险级别: {ToolRiskLevel.CRITICAL}')