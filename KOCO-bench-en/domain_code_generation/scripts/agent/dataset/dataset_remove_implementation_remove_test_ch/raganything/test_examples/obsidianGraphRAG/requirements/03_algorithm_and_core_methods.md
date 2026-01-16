# ObsidianGraphRAG 核心函数描述

# FUNCTION: SimpleRAGAnything.initialize

## 功能概述
`initialize` 是 `SimpleRAGAnything` 类中最核心的初始化方法。它负责在系统启动时完成所有准备工作，包括检查数据库状态、加载本地嵌入模型、配置RAG引擎，并根据情况（新建或加载）正确地实例化 `RAGAnything` 对象，为后续的文档处理和查询做好准备。

## 函数签名

async def initialize(self) -> None

## 输入参数
- `self`: `SimpleRAGAnything` 的实例对象。

(该函数不接收直接参数，而是通过 `self` 访问实例的 `working_dir` 等属性，并通过 `os.getenv` 读取环境变量中的配置。)

## 详细描述
检查现有数据库状态，根据交互模式处理数据库选项，加载本地嵌入模型，配置RAG引擎并实例化RAGAnything对象，最后配置重排器功能。

## 输出
- **无直接返回值**。

## 函数实现
code/src/simple_raganything.py:line 433-570`

## 测试代码
code/test_code/test_initialize.py
