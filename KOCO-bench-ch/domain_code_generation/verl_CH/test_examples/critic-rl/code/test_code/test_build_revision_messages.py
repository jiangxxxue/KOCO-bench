import unittest
import sys
import os
from unittest.mock import MagicMock

# Mock only external dependencies
sys.modules['verl'] = MagicMock()
sys.modules['sandbox_fusion'] = MagicMock()
sys.modules['tenacity'] = MagicMock()

# Add the parent directory (code/) to Python path so we can import ctrl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the actual functions we want to test from critic-rl code
from ctrl.gen.prompt import extract_critique_from_xml
from ctrl.rl.critic_rm import sanitize, desanitize, RewardModelForCritic


class TestBuildRevisionMessages(unittest.TestCase):
    """测试RewardModelForCritic.build_revision_messages函数 - CTRL文档第3节描述的修订提示构建"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建一个简化的RewardModelForCritic实例用于测试
        # 我们不需要完整初始化，只需要能调用build_revision_messages方法
        self.rm = object.__new__(RewardModelForCritic)
    
    def test_build_revision_messages_basic(self):
        """测试基本的修订消息构建功能"""
        init_messages = [
            {"role": "user", "content": "Write a function to add two numbers."},
            {"role": "assistant", "content": "```python\ndef add(a, b):\n    return a + b\n```"}
        ]
        
        critique = """<format>
Analysis:
The solution is correct but lacks error handling.

Improvement suggestions:
Consider adding type checking.

Overall judgment: Correct
</format>"""
        
        result = self.rm.build_revision_messages(init_messages, critique)

        # print(result)

        expected_result = [
            {'role': 'user', 'content': 'Write a function to add two numbers.'}, 
            {'role': 'assistant', 'content': '```python\ndef add(a, b):\n    return a + b\n```'}, 
            {'role': 'user', 'content': 'Analysis:\nThe solution is correct but lacks error handling.\n\nImprovement suggestions:\nConsider adding type checking.\n\nOverall judgment: Correct\n\nPlease finalize your answer accordingly using the same format.'}
        ]

        self.assertEqual(result, expected_result)
    
    def test_input_validation_not_list(self):
        """测试输入验证：init_messages不是列表"""
        init_messages = "not a list"
        critique = "Some critique"
        
        with self.assertRaises(AssertionError):
            self.rm.build_revision_messages(init_messages, critique)
    
    def test_input_validation_wrong_last_role(self):
        """测试输入验证：最后一条消息角色不是assistant"""
        init_messages = [
            {"role": "user", "content": "Question 1"},
            {"role": "user", "content": "Question 2"}  # 错误：应该是assistant
        ]
        critique = "Some critique"
        
        with self.assertRaises(AssertionError):
            self.rm.build_revision_messages(init_messages, critique)
    
    def test_code_sanitization(self):
        """测试代码清理功能 - sanitize函数提取markdown代码块"""
        # 测试包含代码块的内容
        init_messages = [
            {"role": "user", "content": "Write a function"},
            {"role": "assistant", "content": "Here is the code:\n```python\ndef func():\n    pass\n```\nThis is explanation."}
        ]
        critique = "<format>Good</format>"
        
        result = self.rm.build_revision_messages(init_messages, critique)

        # print(result)

        expected_result = [{'role': 'user', 'content': 'Write a function'}, {'role': 'assistant', 'content': '```python\ndef func():\n    pass\n```'}, {'role': 'user', 'content': 'Good\n\nPlease finalize your answer accordingly using the same format.'}]
        
        self.assertEqual(result, expected_result)
    
    def test_code_sanitization_no_code_block(self):
        """测试代码清理 - 没有代码块的情况"""
        init_messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Just plain text without code block"}
        ]
        critique = "<format>Critique</format>"
        
        result = self.rm.build_revision_messages(init_messages, critique)
        
        # print(result)

        expected_result = [{'role': 'user', 'content': 'Question'}, {'role': 'assistant', 'content': 'Just plain text without code block'}, {'role': 'user', 'content': 'Critique\n\nPlease finalize your answer accordingly using the same format.'}]
        # 如果没有代码块，sanitize应该返回原文本
        self.assertEqual(result, expected_result)
    
    def test_critique_extraction_from_xml(self):
        """测试从XML格式中提取批评内容"""
        init_messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "```python\ncode\n```"}
        ]
        
        # 测试包含<format>标签的批评
        critique_with_format = """Some prefix text
<format>
This is the actual critique content.
It has multiple lines.
Overall judgment: Incorrect
</format>
Some suffix text"""
        
        result = self.rm.build_revision_messages(init_messages, critique_with_format)
        
        # print(result)

        expected_result = [{'role': 'user', 'content': 'Question'}, {'role': 'assistant', 'content': '```python\ncode\n```'}, {'role': 'user', 'content': 'This is the actual critique content.\nIt has multiple lines.\nOverall judgment: Incorrect\n\nPlease finalize your answer accordingly using the same format.'}]        
        self.assertEqual(result, expected_result)
        # 验证提取的内容不包含<format>标签和外部文本
        user_content = result[-1]["content"]
        self.assertNotIn("<format>", user_content)
        self.assertNotIn("</format>", user_content)
        self.assertNotIn("Some prefix text", user_content)
        self.assertNotIn("Some suffix text", user_content)
        self.assertIn("actual critique content", user_content)
    
    def test_critique_extraction_no_format_tags(self):
        """测试批评内容没有format标签的情况"""
        init_messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "```python\ncode\n```"}
        ]
        
        # 没有<format>标签的批评
        critique_no_format = "This is plain critique without format tags"
        
        result = self.rm.build_revision_messages(init_messages, critique_no_format)

        # print(result)
        
        expected_result =[{'role': 'user', 'content': 'Question'}, {'role': 'assistant', 'content': '```python\ncode\n```'}, {'role': 'user', 'content': 'This is plain critique without format tags\n\nPlease finalize your answer accordingly using the same format.'}]
        # 如果没有format标签，extract_critique_from_xml应该返回原文本
        self.assertEqual(result, expected_result)
    
    def test_revision_instruction_included(self):
        """测试修订指令被正确添加"""
        init_messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "```python\ncode\n```"}
        ]
        critique = "Your solution needs improvement"
        
        result = self.rm.build_revision_messages(init_messages, critique)
        
        # print(result)

        expected_result = [{'role': 'user', 'content': 'Question'}, {'role': 'assistant', 'content': '```python\ncode\n```'}, {'role': 'user', 'content': 'Your solution needs improvement\n\nPlease finalize your answer accordingly using the same format.'}]
        # 验证修订指令存在
        self.assertEqual(result, expected_result)

    
    def test_dialogue_structure(self):
        """测试完整的对话结构"""
        init_messages = [
            {"role": "user", "content": "Solve problem X"},
            {"role": "assistant", "content": "```python\ndef solution():\n    return 42\n```"}
        ]
        critique = "<format>The solution is incorrect.\nOverall judgment: Incorrect</format>"
        
        result = self.rm.build_revision_messages(init_messages, critique)
        
        # print(result)

        expected_result = [{'role': 'user', 'content': 'Solve problem X'}, {'role': 'assistant', 'content': '```python\ndef solution():\n    return 42\n```'}, {'role': 'user', 'content': 'The solution is incorrect.\nOverall judgment: Incorrect\n\nPlease finalize your answer accordingly using the same format.'}]    
        # 验证对话结构：user -> assistant -> user
        self.assertEqual(result, expected_result)
    
    def test_multiple_code_blocks(self):
        """测试包含多个代码块的情况"""
        init_messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": """First code block:
```python
def func1():
    pass
```
Second code block:
```python
def func2():
    pass
```"""}
        ]
        critique = "Good job"
        
        result = self.rm.build_revision_messages(init_messages, critique)

        # print(result)

        expected_result =[{'role': 'user', 'content': 'Question'}, {'role': 'assistant', 'content': '```python\ndef func1():\n    pass\n```'}, {'role': 'user', 'content': 'Good job\n\nPlease finalize your answer accordingly using the same format.'}]
        
        # sanitize函数应该只提取第一个代码块
        self.assertEqual(result, expected_result)

    
    def test_whitespace_handling(self):
        """测试空白字符处理"""
        init_messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "  ```python\n  def func():\n      pass\n  ```  "}
        ]
        
        critique = """  <format>
        
        Critique with whitespace
        
        </format>  """
        
        result = self.rm.build_revision_messages(init_messages, critique)

        # print(result)
        
        expected_result =[{'role': 'user', 'content': 'Question'}, {'role': 'assistant', 'content': '```python\ndef func():\n      pass\n```'}, {'role': 'user', 'content': 'Critique with whitespace\n\nPlease finalize your answer accordingly using the same format.'}]
        # extract_critique_from_xml应该strip空白字符
        self.assertEqual(result, expected_result)
        # 批评内容应该被strip



if __name__ == "__main__":
    unittest.main()
