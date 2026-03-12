import asyncio
import cohere
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc


async def main():
    api_key = ""
    gemini_api_key=""

    co = cohere.ClientV2(api_key)

    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        enable_image_processing=False,
        enable_table_processing=True,
        enable_equation_processing=True,
        context_window=5,                  # allow more chunks
        context_mode="chunk",              # chunk-based context
        max_context_tokens=4000,           # more tokens
        include_headers=True,
        include_captions=True,
        context_filter_content_types=["text"]
    )

    # LLM model function
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        # Build messages like OpenAI format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in history_messages:
            messages.append(msg)
        messages.append({
            "role": "user",
            "content": f"Use the following context to answer:\n\n\nQuestion: {prompt}"
        })

        # Call Cohere
        response = await asyncio.to_thread(
            co.chat,
            model="command-a-03-2025",
            messages=messages,
            #documents=[{"text": context}] if context else None
        )

        return response.message.content[0].text

    # Embedding function
    async def embedding_func_async(texts):
        response = co.embed(
            model="embed-v4.0",
            input_type="search_query",
            texts=texts
        )
        return response.embeddings.float

    embedding_func = EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=embedding_func_async
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )

    # Process the document
    await rag.process_document_complete(
        file_path=r"/mnt/c/Users/Zakaria Eisa/RAG-Anything/test2.pdf",
        output_dir="./output",
        parse_method="auto"
    )

    # Query the CV
    text_result = await rag.aquery(
        "ما هي الاعراض الشائعة لكوفيد 19؟",
        mode="hybrid",
        enable_rerank=False,
    )

    print("Text query result:", text_result)


if __name__ == "__main__":
    asyncio.run(main())
