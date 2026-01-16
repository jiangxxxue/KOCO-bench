"""
装饰器工具模块
"""
import functools
import logging
import streamlit as st
from typing import Callable, Any

logger = logging.getLogger(__name__)

def error_handler(show_error: bool = True) -> Callable:
    """
    统一错误处理装饰器
    
    @param {bool} show_error - 是否在UI中显示错误信息
    @returns {Callable} - 装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{func.__name__} 执行失败: {str(e)}")
                if show_error:
                    st.error(f"操作失败: {str(e)}")
                raise
        return wrapper
    return decorator



def log_execution(func: Callable) -> Callable:
    """
    记录函数执行的装饰器
    
    @param {Callable} func - 被装饰的函数
    @returns {Callable} - 装饰器函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger.info(f"开始执行 {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} 执行成功")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} 执行失败: {str(e)}")
            raise
    return wrapper 