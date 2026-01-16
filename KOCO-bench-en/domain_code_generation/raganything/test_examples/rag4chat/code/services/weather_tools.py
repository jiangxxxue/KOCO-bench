"""
天气查询工具
"""
import logging
import requests
import os
import re
import json
from typing import Dict, Any, List, Callable, Optional, Tuple

from config.settings import AMAP_API_KEY

# 配置日志
logger = logging.getLogger(__name__)

class WeatherService:
    """
    基于高德地图API的天气查询服务
    """
    
    # 高德API接口
    WEATHER_API_URL = "https://restapi.amap.com/v3/weather/weatherInfo"
    GEO_API_URL = "https://restapi.amap.com/v3/geocode/geo"
    
    # 1. 初始化天气查询服务
    def __init__(self, api_key: str):
        """
        
        Args:
            api_key: 高德地图API密钥
        """
        self.api_key = api_key
        # logger.info("天气查询服务初始化成功")
    

    # 2. 获取城市编码
    def get_city_code(self, city_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Args:
            city_name: 城市名称
            
        Returns:
            Tuple[Optional[str], Optional[str]]: (adcode, 城市名称) 元组，查询失败则返回(None, None)
        """
        try:
            params = {
                "key": self.api_key,
                "address": city_name,
                "output": "JSON"
            }
            
            response = requests.get(self.GEO_API_URL, params=params)
            data = response.json()
            
            if data["status"] == "1" and data["count"] != "0":
                geocode = data["geocodes"][0]
                return geocode["adcode"], geocode["city"] or geocode["district"]
            else:
                logger.warning(f"未找到城市: {city_name}，API返回: {data}")
                return None, None
                
        except Exception as e:
            logger.error(f"获取城市编码时发生错误: {str(e)}")
            return None, None
    

    # 3. 查询天气信息
    def query_weather(self, city: str, extensions: str = "all") -> Dict[str, Any]:
        """
        
        Args:
            city: 城市名称或编码
            extensions: 气象类型，base-实况天气，all-预报天气（未来3天）
            
        Returns:
            Dict[str, Any]: 天气信息字典
        """
        result = {
            "status": "error",
            "data": None,
            "message": ""
        }
        
        try:
            # 先尝试获取城市编码
            city_code, city_name = city, city
            if not city.isdigit():
                city_code, city_name = self.get_city_code(city)
                
            if not city_code:
                result["message"] = f"无法找到城市: {city}"
                return result
                
            # 请求天气API
            params = {
                "key": self.api_key,
                "city": city_code,
                "extensions": extensions,
                "output": "JSON"
            }
            
            response = requests.get(self.WEATHER_API_URL, params=params)
            data = response.json()
            
            if data["status"] == "1":
                result["status"] = "success"
                result["data"] = data
                
                # 添加一个格式化的简要信息
                if extensions == "base":
                    lives = data.get("lives", [])
                    if lives:
                        weather_info = lives[0]
                        result["summary"] = self._format_current_weather(weather_info, city_name)
                else:
                    forecasts = data.get("forecasts", [])
                    if forecasts and forecasts[0].get("casts"):
                        result["summary"] = self._format_forecast_weather(forecasts[0], city_name)
            else:
                result["message"] = f"天气查询失败，API返回: {data}"
                
            return result
            
        except Exception as e:
            logger.error(f"查询天气时发生错误: {str(e)}")
            result["message"] = f"查询天气时发生错误: {str(e)}"
            return result
            


    # 4. 格式化当前天气信息
    def _format_current_weather(self, weather: Dict[str, Any], city_name: str) -> str:
        return (
            f"{city_name}当前天气: {weather.get('weather')}，气温{weather.get('temperature')}℃，"
            f"湿度{weather.get('humidity')}%，{weather.get('winddirection')}风{weather.get('windpower')}级。"
            f"数据发布时间: {weather.get('reporttime')}"
        )
        

    # 5. 格式化预报天气信息
    def _format_forecast_weather(self, forecast: Dict[str, Any], city_name: str) -> str:
        result = f"{city_name}未来天气预报:\n"
        
        for cast in forecast.get("casts", []):
            date = cast.get("date")
            day_weather = cast.get("dayweather")
            night_weather = cast.get("nightweather")
            day_temp = cast.get("daytemp")
            night_temp = cast.get("nighttemp")
            day_wind = f"{cast.get('daywind')}风{cast.get('daypower')}级"
            night_wind = f"{cast.get('nightwind')}风{cast.get('nightpower')}级"
            
            result += (
                f"{date}: 白天{day_weather} {day_temp}℃ {day_wind}，"
                f"夜间{night_weather} {night_temp}℃ {night_wind}\n"
            )
            
        return result




class WeatherTools:
    """
    基于高德地图API的天气查询工具
    """
    
    def __init__(self, api_key: str):
        """
        初始化天气工具
        
        Args:
            api_key: 高德地图API密钥
        """
        self.weather_service = WeatherService(api_key)
        # logger.info("天气查询工具初始化成功")
    
    def query_weather(self, city: str) -> str:
        """
        查询指定城市的天气预报
        
        Args:
            city: 要查询的城市名称
            
        Returns:
            str: 天气信息
        """
        result = self.weather_service.query_weather(city)
        if result["status"] == "success" and "summary" in result:
            return result["summary"]
        else:
            return f"获取{city}的天气信息失败: {result.get('message', '未知错误')}" 