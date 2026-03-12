import asyncio
import os
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
from google import genai
from google.genai import types

# ========== Gemini client ==========
gemini_api_key = "AIzaSyAmWDfyxz_3_yxx1iKCFOGTfyyeksMz_0Y"
gemini = genai.Client(api_key=gemini_api_key)

# Fix SSL issues
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""


async def main():
    # RAGAnything config
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        context_window=5,
        context_mode="chunk",
        max_context_tokens=4000,
        include_headers=True,
        include_captions=True,
        context_filter_content_types=["text"],
    )

        # ========== LLM Model Function ==========
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], messages=None, **kwargs):
        contents = []

        # System prompt → Gemini doesn't support "system", use "user"
        if system_prompt:
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=system_prompt)]
                )
            )

        # History messages
        for msg in history_messages:
            role = msg.get("role", "user")
            if role not in ["user", "model"]:
                role = "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])]
                )
            )

        # Current messages
        if messages:
            for msg in messages:
                role = msg.get("role", "user")
                if role not in ["user", "model"]:
                    role = "user"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part(text=msg["content"])]
                    )
                )
        else:
            contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: gemini.models.generate_content(
                    model="gemini-2.0-flash-lite",
                    contents=contents,
                ),
            )

            if response and response.candidates:
                return response.candidates[0].content.parts[0].text
            return None

        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):  # Quota exceeded
                print("⚠️ Gemini quota reached. Retrying after delay...")
                await asyncio.sleep(7)
                return await llm_model_func(prompt, system_prompt, history_messages, messages, **kwargs)
            raise


        # ========== Vision Model Function ==========
    async def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
        contents = []

        # System prompt
        if system_prompt:
            contents.append(
                types.Content(
                    role="user",   # Gemini Vision بيسمح user / model بس
                    parts=[types.Part(text=system_prompt)]
                )
            )

        # History messages
        for msg in history_messages:
            role = msg.get("role", "user")
            if role not in ["user", "model"]:
                role = "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])]
                )
            )

        # New messages
        if messages:
            for msg in messages:
                role = msg.get("role", "user")
                if role not in ["user", "model"]:
                    role = "user"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part(text=msg["content"])]
                    )
                )
        elif image_data:
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=prompt),
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/jpeg",
                                data=image_data,
                            )
                        ),
                    ],
                )
            )
        else:
            contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: gemini.models.generate_content(
                    model="gemini-2.0-flash-lite",  # Vision-capable model
                    contents=contents,
                ),
            )

            if response and response.candidates:
                return response.candidates[0].content.parts[0].text
            return None

        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):  # Handle quota
                print("⚠️ Gemini quota reached. Retrying after delay...")
                await asyncio.sleep(7)
                return await vision_model_func(prompt, system_prompt, history_messages, image_data, messages, **kwargs)
            raise


    # ========== Embedding ==========
    async def generate_embedding(model, texts):
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini.models.embed_content(
                model=model,
                contents=[types.Content(parts=[types.Part(text=t)]) for t in texts],
            ),
        )
        return [embedding.values for embedding in response.embeddings]

    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: generate_embedding("gemini-embedding-001", texts),
    )

    # ===== Initialize RAG =====
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # ===== Process a PDF =====
    await rag.process_document_complete(
        file_path=r"/mnt/c/Users/Zakaria Eisa/RAG-Anything/test2.pdf",
        output_dir="./output",
        parse_method="auto",
    )

    # ===== Chat loop =====
    text_result = await rag.aquery(
        "What is covid 19?",
        mode="hybrid",
    )
    print("Text query result:", text_result)


if __name__ == "__main__":
    asyncio.run(main())



'''

    multimodal_result = await rag.aquery_with_multimodal(
        "Explain this formula and its relevance to the document content",
        multimodal_content=[{
            "type": "equation",
            "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
            "equation_caption": "Document relevance probability"
        }],
        mode="hybrid"
    )
    print("Multimodal query result:", multimodal_result)
'''