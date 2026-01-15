import unittest
import sys
import os

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train', 'src'))

# Import the actual function we want to test
from open_r1.grpo import len_reward


class TestLenReward(unittest.TestCase):
    """测试len_reward函数 - CANOE文档第4节描述的长答案长度约束"""
    
    def test_len_reward_correct_length(self):
        """测试长度在正确范围内的情况"""
        # context长度为100，long_answer长度为50（在20%-80%范围内）
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 50 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\n问题内容"]
        
        # 调用真实的len_reward函数
        rewards = len_reward(completions, problem=problem)
        
        # 验证输出
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)  # 长度在范围内
    
    def test_len_reward_too_short(self):
        """测试长答案太短的情况"""
        # context长度为100，long_answer长度为10（小于20%）
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 10 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\n问题内容"]
        
        rewards = len_reward(completions, problem=problem)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # 长度太短
    
    def test_len_reward_too_long(self):
        """测试长答案太长的情况"""
        # context长度为100，long_answer长度为90（大于80%）
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 90 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\n问题内容"]
        
        rewards = len_reward(completions, problem=problem)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # 长度太长
    
    def test_len_reward_boundary_min(self):
        """测试最小边界情况"""
        # context长度为100，long_answer长度为21（刚好大于20%）
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 21 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\n问题内容"]
        
        rewards = len_reward(completions, problem=problem)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)  # 刚好在范围内
    
    def test_len_reward_boundary_max(self):
        """测试最大边界情况"""
        # context长度为100，long_answer长度为79（刚好小于80%）
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 79 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\n问题内容"]
        
        rewards = len_reward(completions, problem=problem)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)  # 刚好在范围内
    
    def test_len_reward_exact_min_boundary(self):
        """测试恰好等于最小边界的情况"""
        # context长度为100，long_answer长度为20（恰好等于20%）
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 20 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\n问题内容"]
        
        rewards = len_reward(completions, problem=problem)
        
        # 由于是严格大于，恰好等于20%应该不满足
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_len_reward_exact_max_boundary(self):
        """测试恰好等于最大边界的情况"""
        # context长度为100，long_answer长度为80（恰好等于80%）
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 80 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\n问题内容"]
        
        rewards = len_reward(completions, problem=problem)
        
        # 由于是严格小于，恰好等于80%应该不满足
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_len_reward_missing_context_tag(self):
        """测试缺少context标签的情况"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 50 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["没有context标签的问题"]
        
        rewards = len_reward(completions, problem=problem)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # 无法提取context，返回0
    
    def test_len_reward_missing_long_answer_tag(self):
        """测试缺少long_answer标签的情况"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><answer>42</answer>"}],  # 缺少long_answer
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\n问题内容"]
        
        rewards = len_reward(completions, problem=problem)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # 无法提取long_answer，返回0
    
    def test_len_reward_empty_context(self):
        """测试空context的情况"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 50 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context></context>\n\n问题内容"]
        
        rewards = len_reward(completions, problem=problem)
        
        # 空context长度为0，任何长度的long_answer都会超过80%
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_len_reward_empty_long_answer(self):
        """测试空long_answer的情况"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer></long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\n问题内容"]
        
        rewards = len_reward(completions, problem=problem)
        
        # 空long_answer长度为0，小于20%
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_len_reward_whitespace_handling(self):
        """测试空白字符处理"""
        # context和long_answer都包含空白字符
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>  " + "a" * 50 + "  </long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>  " + "x" * 100 + "  </context>\n\n问题内容"]
        
        rewards = len_reward(completions, problem=problem)
        
        # strip()会移除前后空白，但不影响长度计算（因为strip后再计算长度）
        self.assertEqual(len(rewards), 1)
        # 50在(0.2*100, 0.8*100)范围内
        self.assertEqual(rewards[0], 1.0)
    
    def test_len_reward_multiline_content(self):
        """测试多行内容"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a\n" * 25 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "x\n" * 50 + "</context>\n\n问题内容"]
        
        rewards = len_reward(completions, problem=problem)
        
        # 注意：实际代码中的re.search没有使用re.DOTALL标志
        # 这意味着.*?不能匹配换行符，所以包含换行符的内容无法被正则表达式匹配
        # 因此会返回0而不是1
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # 无法匹配多行内容
    
    def test_len_reward_batch_processing(self):
        """测试批量处理"""
        batch_size = 5
        completions = [
            [{"role": "assistant", "content": f"<think>思考</think><long_answer>{'a' * (30 + i * 10)}</long_answer><answer>42</answer>"}]
            for i in range(batch_size)
        ]
        problem = [f"<context>{'x' * 100}</context>\n\n问题{i}" for i in range(batch_size)]
        
        rewards = len_reward(completions, problem=problem)
        
        # 验证批量处理
        self.assertEqual(len(rewards), batch_size)
        # 30, 40, 50, 60, 70 都在(20, 80)范围内
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_len_reward_mixed_batch(self):
        """测试混合结果的批次"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 50 + "</long_answer><answer>42</answer>"}],  # 正确
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 10 + "</long_answer><answer>42</answer>"}],  # 太短
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 90 + "</long_answer><answer>42</answer>"}],  # 太长
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 30 + "</long_answer><answer>42</answer>"}],  # 正确
        ]
        problem = ["<context>" + "x" * 100 + "</context>\n\n问题"] * 4
        
        rewards = len_reward(completions, problem=problem)
        
        self.assertEqual(len(rewards), 4)
        self.assertEqual(rewards[0], 1.0)  # 正确
        self.assertEqual(rewards[1], 0.0)  # 太短
        self.assertEqual(rewards[2], 0.0)  # 太长
        self.assertEqual(rewards[3], 1.0)  # 正确
    
    def test_len_reward_different_context_lengths(self):
        """测试不同context长度的情况"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 25 + "</long_answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 50 + "</long_answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 100 + "</long_answer><answer>42</answer>"}],
        ]
        problem = [
            "<context>" + "x" * 100 + "</context>\n\n问题1",  # context长度100，long_answer 25在(20, 80)内
            "<context>" + "x" * 100 + "</context>\n\n问题2",  # context长度100，long_answer 50在(20, 80)内
            "<context>" + "x" * 200 + "</context>\n\n问题3",  # context长度200，long_answer 100在(40, 160)内
        ]
        
        rewards = len_reward(completions, problem=problem)
        
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 1.0)  # 25在(20, 80)内
        self.assertEqual(rewards[1], 1.0)  # 50在(20, 80)内
        self.assertEqual(rewards[2], 1.0)  # 100在(40, 160)内
    
    def test_len_reward_unicode_characters(self):
        """测试Unicode字符"""
        # 中文字符也应该被正确计数
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "中" * 50 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "文" * 100 + "</context>\n\n问题内容"]
        
        rewards = len_reward(completions, problem=problem)
        
        # 50个中文字符在(20, 80)范围内
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_len_reward_special_characters(self):
        """测试特殊字符"""
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "!@#$%^&*()" * 5 + "</long_answer><answer>42</answer>"}],
        ]
        problem = ["<context>" + "!@#$%^&*()" * 10 + "</context>\n\n问题内容"]
        
        rewards = len_reward(completions, problem=problem)
        
        # 50个字符在(20, 80)范围内
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)


class TestLenRewardIntegration(unittest.TestCase):
    """集成测试：测试len_reward与实际使用场景的集成"""
    
    def test_realistic_scenario(self):
        """测试真实场景"""
        # 模拟一个真实的问题和回答
        context_text = """
        在机器学习中，过拟合是指模型在训练数据上表现很好，但在测试数据上表现较差的现象。
        这通常是因为模型过于复杂，学习了训练数据中的噪声和细节，而不是潜在的模式。
        为了防止过拟合，可以使用正则化、交叉验证、早停等技术。
        """
        
        long_answer_text = """
        过拟合的主要原因是模型复杂度过高。当模型有太多参数时，它可以记住训练数据的每个细节，
        包括噪声。解决方法包括：1) 使用L1或L2正则化来惩罚大的权重；2) 使用dropout来随机丢弃神经元；
        3) 收集更多训练数据；4) 使用交叉验证来选择合适的模型复杂度。
        """
        
        completions = [
            [{"role": "assistant", "content": f"<think>分析问题...</think><long_answer>{long_answer_text}</long_answer><answer>使用正则化和交叉验证</answer>"}],
        ]
        problem = [f"<context>{context_text}</context>\n\n如何防止过拟合？"]
        
        rewards = len_reward(completions, problem=problem)
        
        # 验证长答案长度在合理范围内
        self.assertEqual(len(rewards), 1)
        # 根据实际长度判断
        context_len = len(context_text.strip())
        long_answer_len = len(long_answer_text.strip())
        if 0.2 * context_len < long_answer_len < 0.8 * context_len:
            self.assertEqual(rewards[0], 1.0)
        else:
            self.assertEqual(rewards[0], 0.0)
    
    def test_quality_control_scenario(self):
        """测试质量控制场景"""
        # 模拟不同质量的长答案
        context = "<context>" + "x" * 100 + "</context>\n\n问题"
        
        completions = [
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 50 + "</long_answer><answer>42</answer>"}],  # 好的长度
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 5 + "</long_answer><answer>42</answer>"}],   # 太短，质量差
            [{"role": "assistant", "content": "<think>思考</think><long_answer>" + "a" * 95 + "</long_answer><answer>42</answer>"}],  # 太长，可能冗余
        ]
        problem = [context] * 3
        
        rewards = len_reward(completions, problem=problem)
        
        # 可以用于过滤：只保留长度合适的生成
        good_indices = [i for i, r in enumerate(rewards) if r == 1.0]
        self.assertEqual(good_indices, [0])


if __name__ == "__main__":
    unittest.main()
