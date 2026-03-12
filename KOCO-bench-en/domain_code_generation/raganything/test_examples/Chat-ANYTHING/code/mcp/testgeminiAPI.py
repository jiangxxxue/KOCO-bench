
# ❗️ Make sure to paste your Google API key here
API_KEY = "AIzaSyBFLZ8o7tOCppO60pLBKX9SsGev0FEsavs" 


from google import genai

client = genai.Client(api_key=API_KEY)

result = client.models.embed_content(
        model="gemini-embedding-001",
        contents= [
            "What is the meaning of life?",
            "What is the purpose of existence?",
            "How do I bake a cake?"
        ])

for embedding in result.embeddings:
    print(embedding)