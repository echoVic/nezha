"""
天气和时间相关工具
"""
import random
from datetime import datetime
from ..tools.base import BaseTool

class GetCurrentWeather(BaseTool):
    name = "get_current_weather"
    description = "获取指定城市的当前天气情况"
    arguments = {"location": "城市名称，如北京、上海等"}
    
    def execute(self, location):
        """
        模拟获取城市天气（实际应用中应对接天气API）
        
        Args:
            location: 城市名称
            
        Returns:
            str: 天气描述
        """
        # 模拟天气情况，实际应用中应对接真实天气API
        weather_conditions = ["晴天", "多云", "阴天", "小雨", "大雨", "雷阵雨", "小雪", "大雪"]
        temperatures = list(range(0, 35))
        
        weather = random.choice(weather_conditions)
        temperature = random.choice(temperatures)
        
        return f"{location}今天是{weather}，温度{temperature}°C。"


class GetCurrentTime(BaseTool):
    name = "get_current_time"
    description = "当被问到当前时间、现在是几点、今天是几号、今天星期几、现在日期等时间相关信息时，必须调用此工具获取准确的时间和日期。"
    arguments = {}
    
    def execute(self):
        """
        获取当前系统时间
        
        Returns:
            str: 当前时间的格式化字符串
        """
        print("\n[DEBUG] GetCurrentTime.execute() 被调用！")
        
        try:
            current_datetime = datetime.now()
            formatted_date = current_datetime.strftime('%Y年%m月%d日')
            formatted_time = current_datetime.strftime('%H:%M:%S')
            weekday_names = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
            weekday = weekday_names[current_datetime.weekday()]
            
            result = f"今天是{formatted_date}，{weekday}，现在时间是{formatted_time}。"
            print(f"[DEBUG] GetCurrentTime 工具执行成功，返回结果: {result}")
            return result
        except Exception as e:
            error_msg = f"[ERROR] GetCurrentTime 工具执行失败: {str(e)}"
            print(error_msg)
            return error_msg
