import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

class MockOpenAIServer:
    def __init__(self, host="localhost", port=8008):
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
        self._started = False

    def _create_handler(self):
        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, payload):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

            def do_POST(self):
                try:
                    length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(length).decode()
                    data = json.loads(body) if body else {}
                except Exception:
                    data = {}

                # 提取 prompt（用于日志和判断）
                prompt = ""
                system_prompt = ""
                messages = data.get("messages", [])
                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if role == "system" and isinstance(content, str):
                        system_prompt = content
                    elif role == "user" and isinstance(content, str):
                        prompt = content
                        break  # 只取第一个 user 消息
                # if len(prompt)>0:
                #     print(f"🔍 Received prompt : {prompt}...")

                # 判断是否为 LightRAG 知识图谱提取任务
                is_lightrag_extraction = (
                        "Knowledge Graph Specialist" in system_prompt
                        and "entity<|#|>" in system_prompt
                        and "<|COMPLETE|>" in prompt
                )

                if "/chat/completions" in self.path:
                    # 默认响应
                    response_content = "Mock Response: This is a fake OpenAI chat completion."

                    prompt_lower = prompt.lower()

                    if is_lightrag_extraction:
                        # ✅ 正确响应 LightRAG 图谱提取请求
                        response_content = (
                            "entity<|#|>Test User<|#|>Person<|#|>Author of the test document.\n"
                            "entity<|#|>Test Document<|#|>Content<|#|>A functional test file containing multilingual text, symbols, and formatting examples.\n"
                            "entity<|#|>Hello World<|#|>Content<|#|>A classic programming example phrase included in the document.\n"
                            "<|COMPLETE|>"
                        )
                    elif "keyword" in prompt_lower or "extract" in prompt_lower:
                        # 兼容关键词提取（返回 JSON）
                        response_content = json.dumps({
                            "low_level_keywords": ["mock", "test", "document"],
                            "high_level_keywords": ["rag", "mockserver"]
                        })
                    # 注意：不再需要 "entity" 分支，因为已被 is_lightrag_extraction 覆盖

                    resp = {
                        "id": "mock-chatcmpl-123",
                        "object": "chat.completion",
                        "created": 1234567890,
                        "model": "gpt-4o-mock",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": response_content},
                            "finish_reason": "stop"
                        }],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
                    }
                    self._send_json(resp)
                    return

                elif "/embeddings" in self.path:
                    inputs = data.get("input", [])
                    if isinstance(inputs, str):
                        inputs = [inputs]
                    embeddings = [{"object": "embedding", "embedding": [0.1] * 1024, "index": i} for i in
                                  range(len(inputs))]
                    resp = {
                        "object": "list",
                        "data": embeddings,
                        "model": "text-embedding-3-large-mock",
                        "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)}
                    }
                    self._send_json(resp)
                    return

                self.send_response(404)
                self._send_json({"error": "Not found"})

            def log_message(self, format, *args):
                pass

        return Handler

    def start(self):
        if self._started:
            return
        handler_class = self._create_handler()
        self.server = HTTPServer((self.host, self.port), handler_class)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        self._started = True
        time.sleep(0.1)
        print(f"🔥 Mock OpenAI Server started at http://{self.host}:{self.port}")

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=1)
            self._started = False
            print("🛑 Mock OpenAI Server stopped.")
