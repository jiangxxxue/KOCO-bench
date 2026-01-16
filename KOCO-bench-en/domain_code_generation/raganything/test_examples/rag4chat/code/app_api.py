from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import time
import random
import logging
# 消息格式
# {
#     "model": "Qwen3",
#     "modelId": "XXXXXX",
#     "messages": [
#         {
#             "chatCode": "1ywj4ox2uvb400",
#             "content": "多通道语义图像",
#             "role": "user",
#             "msgTime": 1754528942728,
#             "contentType": "1",
#             "chatFiles": [],
#             "chatModelCode": "Qwen3",
#             "tokenCount": 0
#         }
#     ],
#     "temperature": 1,
#     "isRAG": true,
#     "stream": true,
#     "similarity": 0.5,
#     "apiKey": "sk-6Xc1YUnHfPn98EbWXOE4zA"
# }

# 调用格式
# http://192.10.220.191:19006/health
# http://192.10.220.191:19006/v1/chat/completions



app = Flask(__name__)

# 配置 CORS
CORS(app, resources={
    r"/v1/chat/completions": {
        "origins": ["http://localhost:*", "https://jgyw.crfsdi.com.cn:*", "*"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# get_response.py
import asyncio
from get_response import agent_response

# 创建全局的响应处理器实例
response_handler = agent_response()

def generate_ai_response(user_message):
    """
    生成AI响应的同步函数
    user_message: 用户输入的消息
    返回: AI的响应文本
    """
    try:
        # 运行异步函数并获取结果
        print("开始执行异步任务...")
        response = asyncio.run(response_handler.run(user_message))
        return response
    except Exception as e:
        print(f"生成AI响应时出错: {e}")
        return "抱歉，处理您的请求时出现了问题。"

# 异步版本（如果在异步环境中使用）
async def generate_ai_response_async(user_message):
    """
    生成AI响应的异步函数
    user_message: 用户输入的消息
    返回: AI的响应文本
    """
    try:
        response = await response_handler.run(user_message)
        return response
    except Exception as e:
        print(f"生成AI响应时出错: {e}")
        return "抱歉，处理您的请求时出现了问题。"

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    """处理聊天完成请求"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400
        
        logger.info(f'收到请求: {json.dumps(data, ensure_ascii=False)}')
        
        # 提取必要参数
        model = data.get('model', 'default-model')
        model = "qwen3:8b"
        messages = data.get('messages', [])
        stream = data.get('stream', False)
        temperature = data.get('temperature', 1.0)
        api_key = data.get('apiKey', '')
        isRAG = data.get('isRAG', False)
        sim = data.get('similarity', 0.5)
        is_summary = data.get('is_summary', False)
        # print(is_summary)

        response_handler.update(isRAG, model, sim)
        response_handler.human_content = None
        
        # 验证必要参数
        if not messages:
            return jsonify({"error": "Messages are required"}), 400
        
        # 解析用户消息（从 messages 数组中提取第一条用户消息）
        user_message = ""
        chatCode = "",
        for msg in messages:
            if msg.get('role') == 'user':
                user_message = msg.get('content', '')
                chatCode = msg.get('chatCode', "")
                break
        
        if not user_message:
            return jsonify({"error": "No user message found"}), 400
        
        logger.info(f'用户消息: {user_message}')
        
        # 可选：验证 API Key
        if api_key and not api_key.startswith('sk-'):
            logger.warning(f'无效的 API Key: {api_key}')
            # 可以选择是否拒绝请求
        
        if stream:
            # 流式响应
            def generate_stream():
                if is_summary:
                    response_text = response_handler.get_history_summary()
                else:
                    response_text = generate_ai_response(user_message)
                human = 0 if response_handler.human_content is None else 1

                # 逐字符发送响应
                for i, char in enumerate(response_text):
                    chunk = {
                        "id": f"chatcmpl-{int(time.time()*1000)}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "chatCode": chatCode,
                        "human": human,
                        "choices": [{
                            
                            "delta": {
                                "content": char
                            },
                            "index": 0,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    time.sleep(0.01)  # 控制发送速度
                
                # 发送结束标记
                end_chunk = {
                    "id": f"chatcmpl-{int(time.time()*1000)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "human": human,
                    "chatCode": chatCode,
                    "choices": [{
                        "delta": {},
                        "index": 0,
                        "finish_reason": "stop"
                    }]
                }
                yield f"{json.dumps(end_chunk, ensure_ascii=False)}\n\n"
                yield "[DONE]\n\n"
            
            return Response(generate_stream(), mimetype='text/event-stream')
        
        else:
            # 非流式响应
            if is_summary:
                response_text = response_handler.get_history_summary()
            else:
                response_text = generate_ai_response(user_message)

            human = 0 if response_handler.human_content is None else 1
            
            response = {
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "human": human,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                        "chatCode": chatCode
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_message),
                    "completion_tokens": len(response_text),
                    "total_tokens": len(user_message) + len(response_text)
                }
            }
            
            logger.info(f'返回响应: {response_text}')
            return jsonify(response)
                
    except json.JSONDecodeError:
        logger.error('JSON 解析错误')
        return jsonify({"error": "Invalid JSON format"}), 400
    except Exception as e:
        logger.error(f'处理请求时出错: {str(e)}')
        return jsonify({"error": "Internal server error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "OK", 
        "timestamp": time.time(),
        "service": "chat-completions-api"
    })

@app.route('/', methods=['GET'])
def home():
    """根路径，返回 API 信息"""
    return jsonify({
        "message": "Chat Completions API Service",
        "version": "1.0.0",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "health_check": "/health"
        }
    })

# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # print(generate_ai_response("hello world"))
    app.run(
        host='0.0.0.0', 
        port=19006, 
        debug=True,
        threaded=True  # 支持多线程处理并发请求
    )
    # http://192.10.220.191:19006/v1/chat/completions