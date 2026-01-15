# -*- coding: utf-8 -*-
from langchain.schema import Document
from services.vector_store import VectorStoreService

# 假设已经有一个VectorStoreService实例
# service = VectorStoreService()

# # 创建一些示例文档
# documents = [
#     Document(page_content="这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。它包含了多个句子，每个句子都可能成为分割点。"),
#     Document(page_content="这是另一个文档，同样需要被分割以适应后续的处理。白驹过隙，转眼间硕士三年学习生涯就快要结束，至此，我已在华中科技大学度过了七年时光。	本科二年级时，因为C语言课程设计和彭刚老师幸运相识，硕士时更是得到彭刚老师赏识，拜入门下开始新的学习生涯。从大四开始，到硕士毕业，彭刚老师不仅在研究学习上指导我们，还在为人处事方面为我们树立榜样。在实验室相关项目的研究工作中，我们跟随彭刚老师的步伐，坚定推进相关研究工作，增长了自己的能力与见识。彭刚老师的指导，为我们迈入社会，并成为顶天立地的成年人打下了坚实的基础。在论文研究与撰写工作期间，彭刚老师也相当负责，为我提供了大量的帮助，不辞辛苦地抓论文细节，力求打造高水平的毕业论文。为此，我非常感谢彭刚老师的知遇之恩，这些年在彭老师门下的学习生活我没齿难忘，衷心祝愿彭刚老师在学术上更进一步。在硕士期间，我认识了实验室的各位同学。在项目与课题上，我们合作推进工作的顺利完成。感谢李剑峰、谭则杰、杨进、虎璐等师兄在研究工作中为我提供的指导，感谢关尚宾、万少威等同级在项目和课题中的同心戮力，感谢李志勇、罗新鹏等师弟在相关研究工作中的交流与合作，并特别感谢来自钟胜老师门下的董雷震同学为我的论文相关研究提供的大量专业指导。我们的缘分仍未尽于此，希望即使是踏入职场之后我们也能常常相聚。同时还要感谢北京铁道工程技术研究所的动车巡检项目组同事们，在项目上我们携手攻克了诸多难题，项目的成功推进也是我的论文写作的重要前提，并且各位同事所提供的相关资料也为我的论文内容添砖加瓦。感谢我的父母与家人在我研究生学习期间所提供的支持与帮助。感谢我的爱人近十年以来的长久陪伴，为我的情感提供一处温柔的港湾，走过了学生生涯，我们的故事才刚刚开始。最后感谢各位盲审专家以及答辩组老师为我的论文提供宝贵的指导意见，给我的论文工作画上一个完美的句号")
# ]

# # 使用text_splitter.split_documents方法分割文档
# split_documents = service.text_splitter.split_documents(documents)

# # 打印分割后的文档
# for i, doc in enumerate(split_documents):
#     print(f"Split Document {i+1}: {doc.page_content}")

import logging
from services.vector_store import VectorStoreService
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 创建一个VectorStoreService实例，使用临时目录
    index_dir = 'temp_index'
    service = VectorStoreService(index_dir=index_dir)

    # 创建一些示例文档
    documents = [
    Document(page_content="这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。这是一个很长的文本示例，用于展示文本分割器的功能。它包含了多个句子，每个句子都可能成为分割点。"),
    Document(page_content="这是另一个文档，同样需要被分割以适应后续的处理。白驹过隙，转眼间硕士三年学习生涯就快要结束，至此，我已在华中科技大学度过了七年时光。	本科二年级时，因为C语言课程设计和彭刚老师幸运相识，硕士时更是得到彭刚老师赏识，拜入门下开始新的学习生涯。从大四开始，到硕士毕业，彭刚老师不仅在研究学习上指导我们，还在为人处事方面为我们树立榜样。在实验室相关项目的研究工作中，我们跟随彭刚老师的步伐，坚定推进相关研究工作，增长了自己的能力与见识。彭刚老师的指导，为我们迈入社会，并成为顶天立地的成年人打下了坚实的基础。在论文研究与撰写工作期间，彭刚老师也相当负责，为我提供了大量的帮助，不辞辛苦地抓论文细节，力求打造高水平的毕业论文。为此，我非常感谢彭刚老师的知遇之恩，这些年在彭老师门下的学习生活我没齿难忘，衷心祝愿彭刚老师在学术上更进一步。在硕士期间，我认识了实验室的各位同学。在项目与课题上，我们合作推进工作的顺利完成。感谢李剑峰、谭则杰、杨进、虎璐等师兄在研究工作中为我提供的指导，感谢关尚宾、万少威等同级在项目和课题中的同心戮力，感谢李志勇、罗新鹏等师弟在相关研究工作中的交流与合作，并特别感谢来自钟胜老师门下的董雷震同学为我的论文相关研究提供的大量专业指导。我们的缘分仍未尽于此，希望即使是踏入职场之后我们也能常常相聚。同时还要感谢北京铁道工程技术研究所的动车巡检项目组同事们，在项目上我们携手攻克了诸多难题，项目的成功推进也是我的论文写作的重要前提，并且各位同事所提供的相关资料也为我的论文内容添砖加瓦。感谢我的父母与家人在我研究生学习期间所提供的支持与帮助。感谢我的爱人近十年以来的长久陪伴，为我的情感提供一处温柔的港湾，走过了学生生涯，我们的故事才刚刚开始。最后感谢各位盲审专家以及答辩组老师为我的论文提供宝贵的指导意见，给我的论文工作画上一个完美的句号")
]

    # 调用create_vector_store方法
    logger.info("开始创建向量存储")
    vector_store = service.create_vector_store(documents)
    if vector_store:
        logger.info(f"向量存储创建成功，包含 {len(documents)} 个文档块")
    else:
        logger.warning("向量存储创建失败")
    


    # 清理临时目录
    # import shutil
    # shutil.rmtree(index_dir, ignore_errors=True)


if __name__ == '__main__':
    # main()
    import faiss
    import numpy as np

    def print_faiss_index_contents(index_path, id_to_content_map):
        """
        打印FAISS索引中的内容和对应的向量。

        :param index_path: FAISS索引文件的路径。
        :param id_to_content_map: 一个字典，映射向量ID到它们对应的原始内容。
        """
        # 加载FAISS索引
        index = faiss.read_index(index_path)
        
        # 获取索引中存储的所有向量
        xb = index.reconstruct_n(0, index.ntotal)
        
        # 获取索引中存储的所有向量的ID
        ids = np.arange(index.ntotal)
        
        # 打印向量和它们的ID以及对应的内容
        for i in range(len(xb)):
            # 假设id_to_content_map字典中存储了ID到内容的映射
            content = id_to_content_map.get(ids[i], "Content not found")
            print(f"ID: {ids[i]}, Vector shape: {xb[i].shape}, Content: {content[:100]}...")  # 打印内容的前100个字符

    # 指定索引文件路径
    index_path = r"temp_index/index.faiss"

    # 假设我们有一个ID到内容的映射字典
    id_to_content_map = {
        0: "这是第一个文档的内容...",
        1: "这是第二个文档的内容...",
        # 更多ID到内容的映射...
    }

    # 调用函数打印FAISS索引内容
    print_faiss_index_contents(index_path, id_to_content_map)