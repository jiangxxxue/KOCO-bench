"""
对话历史管理类
"""
import json
import os
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
from config.settings import HISTORY_FILE, MAX_HISTORY_TURNS

class ChatHistoryManager:
    """
    对话历史管理器类
    """
    def __init__(self):
        """初始化对话历史管理器"""
        self.history: List[Dict] = self.load_history()
    
    # 1. 从文件加载对话历史
    def load_history(self) -> List[Dict]:
        """
        Returns:
            List[Dict]: 对话历史记录列表
        """
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"加载对话历史时出错: {str(e)}")
        return []
    
    # 2. 保存对话历史到文件
    def save_history(self) -> None:
        try:
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存对话历史时出错: {str(e)}")
    
    # 3. 添加新消息到历史记录
    def add_message(self, role: str, content: str) -> None:
        """
        Args:
            role (str): 消息角色 ('user' 或 'assistant')
            content (str): 消息内容
        """
        self.history.append({"role": role, "content": content})
        self.save_history()
    
    # 4. 清空对话历史
    def clear_history(self) -> None:
        self.history = []
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
    
    # 5. 获取格式化的对话历史
    def get_formatted_history(self, max_turns: int = MAX_HISTORY_TURNS) -> str:
        """
        Args:
            max_turns (int): 最大保留的对话轮数
            
        Returns:
            str: 格式化后的对话历史
        """
        if not self.history:
            return ""
        
        recent_history = self.history[-max_turns*2:] if len(self.history) > max_turns*2 else self.history
        
        formatted_history = "以下是之前的对话历史：\n"
        for msg in recent_history:
            if msg["role"] == "user":
                role = "用户"
            elif msg["role"] == "assistant":
                role = "助手"
            else:
                continue
            
            formatted_history += f"{role}: {msg['content']}\n"
        
        return formatted_history
    
    # 6. 导出对话历史为CSV文件
    def export_to_csv(self) -> Optional[bytes]:
        """
        导出对话历史为CSV文件
        
        Returns:
            Optional[bytes]: CSV文件内容，如果导出失败则返回None
        """
        try:
            df = pd.DataFrame(self.history)
            return df.to_csv(index=False).encode('utf-8')
        except Exception as e:
            print(f"导出对话历史时出错: {str(e)}")
            return None
    
    # 7. 获取对话历史统计信息
    def get_stats(self) -> Dict[str, int]:
        """
        获取对话历史统计信息
        
        Returns:
            Dict[str, int]: 包含总消息数和用户消息数的字典
        """
        total_messages = len(self.history)
        user_messages = sum(1 for msg in self.history if msg["role"] == "user")
        return {
            "total_messages": total_messages,
            "user_messages": user_messages
        } 