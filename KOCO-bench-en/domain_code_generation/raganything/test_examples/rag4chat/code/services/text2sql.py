"""
数据库查询工具
"""
import logging
import sqlite3
import json
from typing import Dict, Any, List, Callable, Optional, Tuple
from langchain_community.utilities import SQLDatabase




# 配置日志
logger = logging.getLogger(__name__)

class DatabaseService:
    """
    基于SQLite的数据库查询服务
    """
    
    # 1. 初始化数据库查询服务
    def __init__(self, db_path: str):
        """
        初始化数据库查询服务
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        logger.info("数据库查询服务初始化成功")
    
    # 2. 查询数据库表结构
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        查询指定表的结构
        
        Args:
            table_name: 表名
            
        Returns:
            Dict[str, Any]: 表结构信息
        """
        result = {
            "status": "error",
            "data": None,
            "message": ""
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 查询表结构
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            if columns:
                result["status"] = "success"
                result["data"] = [{"name": col[1], "type": col[2]} for col in columns]
            else:
                result["message"] = f"表 {table_name} 不存在或无列信息"
            
            conn.close()
        except Exception as e:
            logger.error(f"查询表结构时发生错误: {str(e)}")
            result["message"] = f"查询表结构时发生错误: {str(e)}"
        
        return result
    
    # 3. 查询数据库数据
    def query_data(self, table_name: str, query: str) -> Dict[str, Any]:
        """
        查询指定表的数据
        
        Args:
            table_name: 表名
            query: 查询条件（SQL语句片段）
            
        Returns:
            Dict[str, Any]: 查询结果
        """
        result = {
            "status": "error",
            "data": None,
            "message": ""
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 构造查询语句
            full_query = f"SELECT * FROM {table_name} WHERE {query}"
            cursor.execute(full_query)
            rows = cursor.fetchall()
            
            if rows:
                result["status"] = "success"
                result["data"] = rows
            else:
                result["message"] = f"未找到符合条件的数据"
            
            conn.close()
        except Exception as e:
            logger.error(f"查询数据时发生错误: {str(e)}")
            result["message"] = f"查询数据时发生错误: {str(e)}"
        
        return result

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

class DatabaseTools:
    """
    数据库查询工具
    """
    
    def __init__(self, db_path: str):
        """
        初始化数据库工具
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_service = DatabaseService(db_path)
        self.db = SQLDatabase.from_uri('sqlite:///D:\\adavance\\bigmodel\\2.原创案例：Agentic RAG智能问答系统Agent\\chinook.db')
        self.generate_query_system_prompt = """
        你是一个设计用于与SQL数据库交互的智能体。
        给定一个输入问题，创建一个语法正确的{dialect}查询来运行，
        然后查看查询结果并返回答案。除非用户明确指定他们希望获取的示例数量，
        否则始终将查询限制为最多{top_k}个结果。

        你可以按相关列对结果进行排序，以返回数据库中最有趣的示例。
        永远不要查询特定表的所有列，只询问与问题相关的列。

        不要对数据库执行任何DML语句（INSERT、UPDATE、DELETE、DROP等）。
        """.format(
            dialect=self.db.dialect,
            top_k=5,
        )

        self.query_check_system = """您是一位注重细节的SQL专家。
        请仔细检查SQLite查询中的常见错误，包括：
        - Using NOT IN with NULL values
        - Using UNION when UNION ALL should have been used
        - Using BETWEEN for exclusive ranges
        - Data type mismatch in predicates
        - Properly quoting identifiers
        - Using the correct number of arguments for functions
        - Casting to the correct data type
        - Using the proper columns for joins

        如果发现上述任何错误，请重写查询。如果没有错误，请原样返回查询语句。

        检查完成后，您将调用适当的工具来执行查询。"""

        self.check_query_system_prompt = """
        你是一个具有高度注意细节能力的SQL专家。
        仔细检查{dialect}查询中的常见错误，包括：
        - 在NULL值上使用NOT IN
        - 应该使用UNION ALL时却使用了UNION
        - 在独占范围上使用BETWEEN
        - 谓词中的数据类型不匹配
        - 正确引用标识符
        - 为函数使用正确数量的参数
        - 转换为正确的数据类型
        - 使用正确的列进行连接

        如果存在上述任何错误，请重写查询。如果没有错误，
        只需重现原始查询。

        在运行此检查后，你将调用适当的工具来执行查询。
        """.format(dialect=self.db.dialect)

        logger.info("数据库查询工具初始化成功")
    
    def query_table_schema(self, table_name: str) -> str:
        """
        查询指定表的结构
        
        Args:
            table_name: 表名
            
        Returns:
            str: 表结构信息
        """
        result = self.db_service.get_table_schema(table_name)
        if result["status"] == "success":
            schema_info = json.dumps(result["data"], ensure_ascii=False, indent=2)
            return f"表 {table_name} 的结构信息：\n{schema_info}"
        else:
            return f"获取表 {table_name} 的结构失败: {result.get('message', '未知错误')}"
    
    def query_table_data(self, table_name: str, query: str) -> str:
        """
        查询指定表的数据
        
        Args:
            table_name: 表名
            query: 查询条件（SQL语句片段）
            
        Returns:
            str: 查询结果
        """
        result = self.db_service.query_data(table_name, query)
        if result["status"] == "success":
            data_info = json.dumps(result["data"], ensure_ascii=False, indent=2)
            return f"查询结果：\n{data_info}"
        else:
            return f"查询数据失败: {result.get('message', '未知错误')}"
    
    def list_tables(self) -> str:
        """输入是一个空字符串, 返回数据库中的所有：以逗号分隔的表名字列表"""
        return ", ".join(self.db.get_usable_table_names())  #   ['emp': “这是一个员工表，”, '']
        
    def query_text2sql(self, query: str) -> str:
        """
        执行SQL查询并返回结果。
        如果查询不正确，将返回错误信息。
        如果返回错误，请重写查询语句，检查后重试。

        Args:
            query (str): 要执行的SQL查询语句

        Returns:
            str: 查询结果或错误信息
        """
        result = self.db.run_no_throw(query)  # 执行查询（不抛出异常）
        if not result:
            return "错误: 查询失败。请修改查询语句后重试。"
        return result
        

import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数据库路径
DATABASE_PATH = r"D:\adavance\bigmodel\2.原创案例：Agentic RAG智能问答系统Agent\chinook.db"

# 初始化数据库工具
db_tools = DatabaseTools(db_path=DATABASE_PATH)
print(db_tools.list_tables())

# 测试查询表结构
def test_query_table_schema():
    table_name = "artists"  # 假设数据库中有这个表
    result = db_tools.query_table_schema(table_name)
    print(f"查询表结构结果：\n{result}")

# 测试查询表数据
def test_query_table_data():
    table_name = "test_1"  # 假设数据库中有这个表
    query = "department = 'Sales'"  # 假设表中有department字段
    result = db_tools.query_table_data(table_name, query)
    print(f"查询表数据结果：\n{result}")

# 主函数
if __name__ == "__main__":
    logger.info("开始测试数据库查询工具")
    
    # 测试查询表结构
    test_query_table_schema()
    
    # 测试查询表数据
    test_query_table_data()
    
    logger.info("测试完成")