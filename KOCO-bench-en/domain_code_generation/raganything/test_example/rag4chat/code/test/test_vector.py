import unittest
from services.vector_store import VectorStoreService
from langchain.schema import Document

class TestVectorStoreService(unittest.TestCase):
    def setUp(self):
        # 创建一个VectorStoreService实例，使用临时目录
        self.index_dir = 'temp_index'
        self.service = VectorStoreService(index_dir=self.index_dir)

    def tearDown(self):
        # 测试完成后，删除临时目录
        import shutil
        shutil.rmtree(self.index_dir, ignore_errors=True)

    def test_create_and_load_vector_store(self):
        # 创建一些测试文档
        documents = [Document(page_content="This is a test document.")]
        
        # 创建向量存储
        vector_store = self.service.create_vector_store(documents)
        self.assertIsNotNone(vector_store)
        
        # 加载向量存储
        loaded_vector_store = self.service.load_vector_store()
        self.assertIsNotNone(loaded_vector_store)

    def test_search_documents(self):
        # 创建一些测试文档
        documents = [Document(page_content="This is a test document.")]
        
        # 创建向量存储
        self.service.create_vector_store(documents)
        
        # 搜索文档
        results = self.service.search_documents("test document", threshold=0.0)
        self.assertEqual(len(results), 1)

    def test_add_document(self):
        # 添加一个新文档
        result = self.service.add_document("New document content")
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()