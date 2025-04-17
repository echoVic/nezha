import os
from openai import OpenAI
import json

print("开始测试火山引擎 API 连接...")

# 初始化 API 客户端
api_key = "1ddfaee1-1350-46b0-ab87-2db988d24d4b"
base_url = "https://ark.cn-beijing.volces.com/api/v3"
model_id = "ep-20250417144747-rgffm"

print(f"API 密钥: {api_key[:4]}...{api_key[-4:]}")
print(f"API 端点: {base_url}")
print(f"模型 ID: {model_id}")

client = OpenAI(
    base_url=base_url,
    api_key=api_key
)

# 使用纯文本消息
try:
    print("\n发送纯文本请求...")
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": "你好，请简单介绍一下自己。"}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    print("\n成功收到响应!")
    print(f"响应内容: {response.choices[0].message.content}")
except Exception as e:
    print(f"\n请求失败: {e}")
    print(f"错误类型: {type(e).__name__}")
    
    # 尝试提取更多错误信息
    if hasattr(e, 'response'):
        try:
            error_detail = json.loads(e.response.text)
            print(f"错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
        except:
            print(f"原始错误响应: {e.response.text if hasattr(e.response, 'text') else '无法获取'}")

print(response.choices[0])