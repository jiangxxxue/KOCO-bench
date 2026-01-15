
## 3. app.py


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

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import json
import time
import logging
import os
import asyncio
from threading import Thread
from queue import Queue
import re

app = Flask(__name__)

# 从环境变量获取配置
FLASK_ENV = os.environ.get('FLASK_ENV', 'development')

# 配置 CORS - 允许所有来源（生产环境建议具体配置）
CORS(app, resources={
    r"/v1/*": {
        "origins": ["*"],  # 允许所有来源
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": True
    }
})

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# get_response.py
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

def process_image_content(chatFiles, logger):
    """
    优先使用OCR处理图片内容，失败时回退到视觉模型
    顺序：PaddleOCR → Tesseract → 视觉模型
    """
    try:
        import requests
        import re
        import base64
        import io
        from PIL import Image
        
        # 1. 初始化PaddleOCR
        paddle_ocr_engine = None
        try:
            from paddleocr import PaddleOCR
            # 尝试不同版本的参数
            try:
                # 新版本参数
                paddle_ocr_engine = PaddleOCR(
                    use_angle_cls=True, 
                    lang='ch', 
                    use_gpu=False
                )
            except TypeError:
                # 旧版本参数
                try:
                    paddle_ocr_engine = PaddleOCR(
                        use_angle_cls=True, 
                        lang='ch', 
                        use_gpu=False,
                        det_model_dir=None,
                        rec_model_dir=None
                    )
                except TypeError:
                    # 最简参数
                    paddle_ocr_engine = PaddleOCR(
                        use_angle_cls=True, 
                        lang='ch'
                    )
            
            logger.info("✅ PaddleOCR初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ PaddleOCR初始化失败: {e}")
            paddle_ocr_engine = None
        
        # 2. 初始化Tesseract OCR作为备用
        tesseract_available = False
        try:
            import pytesseract
            # 测试Tesseract是否可用
            pytesseract.image_to_string(Image.new('RGB', (10, 10), color='white'))
            tesseract_available = True
            logger.info("✅ Tesseract OCR可用")
        except Exception as e:
            logger.warning(f"⚠️ Tesseract OCR不可用: {e}")
            tesseract_available = False
        
        if not chatFiles or len(chatFiles) == 0:
            return "没有上传任何图片"
        
        total_images = len(chatFiles)
        image_descriptions = []
        
        # 为每张图片进行识别
        for index, image_base64 in enumerate(chatFiles, 1):
            try:
                # 预处理base64数据
                clean_base64 = image_base64
                if isinstance(image_base64, str) and ',' in image_base64:
                    clean_base64 = image_base64.split(',')[1]
                
                ocr_text = ""
                ocr_method = ""
                ocr_success = False
                
                # 1. 优先使用PaddleOCR
                if paddle_ocr_engine is not None:
                    try:
                        logger.info(f"使用PaddleOCR处理图片{index}...")
                        image_data = base64.b64decode(clean_base64)
                        image = Image.open(io.BytesIO(image_data))
                        
                        # 转换为numpy数组（PaddleOCR需要）
                        import cv2
                        import numpy as np
                        img_array = np.array(image)
                        if len(img_array.shape) == 3:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        
                        # 执行OCR
                        ocr_result = paddle_ocr_engine.ocr(img_array, cls=True)
                        
                        # 提取文本
                        texts = []
                        if ocr_result is not None:
                            for line in ocr_result:
                                if line is not None:
                                    for word_info in line:
                                        if isinstance(word_info, list) and len(word_info) > 1:
                                            # 兼容不同版本的返回格式
                                            if isinstance(word_info[1], list):
                                                text = word_info[1][0]
                                            else:
                                                text = str(word_info[1])
                                            if text.strip():  # 过滤空文本
                                                texts.append(text.strip())
                        
                        ocr_text = ' '.join(texts).strip()
                        ocr_success = bool(ocr_text and len(ocr_text) > 3)
                        ocr_method = "PaddleOCR"
                        
                        if ocr_success:
                            logger.info(f'图片{index} PaddleOCR识别成功，文本长度: {len(ocr_text)}')
                        else:
                            logger.warning(f'图片{index} PaddleOCR识别结果为空或太短')
                            
                    except Exception as paddle_error:
                        logger.warning(f"图片{index} PaddleOCR识别失败: {paddle_error}")
                        import traceback
                        logger.debug(f"PaddleOCR错误详情: {traceback.format_exc()}")
                        ocr_success = False
                
                # 2. PaddleOCR失败，尝试Tesseract
                if not ocr_success and tesseract_available:
                    try:
                        logger.info(f"使用Tesseract处理图片{index}...")
                        image_data = base64.b64decode(clean_base64)
                        image = Image.open(io.BytesIO(image_data))
                        
                        # 执行OCR识别（支持中英文）
                        ocr_text = pytesseract.image_to_string(image, lang='chi_sim+eng')
                        ocr_text = ' '.join(ocr_text.split()).strip()
                        ocr_success = bool(ocr_text and len(ocr_text) > 3)
                        ocr_method = "Tesseract"
                        
                        if ocr_success:
                            logger.info(f'图片{index} Tesseract识别成功，文本长度: {len(ocr_text)}')
                        else:
                            logger.warning(f'图片{index} Tesseract识别结果为空或太短')
                            
                    except Exception as tesseract_error:
                        logger.warning(f"图片{index} Tesseract识别失败: {tesseract_error}")
                        ocr_success = False
                
                # 3. 如果OCR都失败，回退到视觉模型
                if not ocr_success:
                    logger.info(f"图片{index} OCR失败，回退到视觉模型...")
                    try:
                        # 构造ollama API请求
                        ollama_payload = {
                            "model": "llava:latest",
                            "prompt": "仔细查看这张企业微信截图，提取其中的报错信息、错误代码或关键文本内容。请直接输出识别到的文本，不要添加解释说明。",
                            "images": [clean_base64],
                            "stream": False
                        }
                        
                        ollama_response = requests.post(
                            "http://localhost:11434/api/generate",
                            json=ollama_payload,
                            headers={'Content-Type': 'application/json'},
                            timeout=60
                        )
                        
                        if ollama_response.status_code == 200:
                            response_data = ollama_response.json()
                            vision_text = response_data.get('response', '视觉理解失败')
                            # 清理视觉模型输出
                            vision_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', vision_text)
                            vision_text = re.sub(r'\s+', ' ', vision_text).strip()
                            
                            if vision_text and len(vision_text) > 10:
                                image_descriptions.append(f"图片{index}（视觉识别）：{vision_text}")
                                logger.info(f'图片{index}视觉模型识别成功: {vision_text[:50]}...')
                            else:
                                image_descriptions.append(f"图片{index}：未识别到有效内容")
                                logger.warning(f'图片{index}视觉模型识别结果为空')
                        else:
                            error_msg = f"图片{index}：视觉模型调用失败(状态码:{ollama_response.status_code})"
                            image_descriptions.append(error_msg)
                            logger.error(error_msg)
                            
                    except Exception as vision_error:
                        error_msg = f"图片{index}：视觉模型调用异常({str(vision_error)})"
                        image_descriptions.append(error_msg)
                        logger.error(error_msg)
                else:
                    # OCR成功，使用OCR结果
                    image_descriptions.append(f"图片{index}（{ocr_method}识别）：{ocr_text}")
                    logger.info(f'图片{index} {ocr_method}识别成功: {ocr_text[:50]}...')
                
            except Exception as e:
                error_msg = f"图片{index}：处理过程出现错误({str(e)})"
                image_descriptions.append(error_msg)
                logger.error(error_msg)
        
        # 组合所有图片的描述
        result = f"\n\n【图片信息】用户一共上传了{total_images}个图片"
        if image_descriptions:
            result += "\n" + "\n".join(image_descriptions)
        
        # 最终清理
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        logger.info(f'最终返回结果长度: {len(result)}')
        print("result:", result)
        return result
        
    except Exception as e:
        error_msg = f"图片处理过程出现错误: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg
def process_image_content2(chatFiles, logger):
    """
    调用ollama的llava模型进行多图片识别
    
    Args:
        chatFiles: 包含多个图片base64编码的数组
        logger: 日志记录器
    
    Returns:
        str: 识别后的文本内容描述
    """
    try:
        import requests
        import json
        
        if not chatFiles or len(chatFiles) == 0:
            return "没有上传任何图片"
        
        total_images = len(chatFiles)
        image_descriptions = []
        
        # 为每张图片进行识别
        for index, image_base64 in enumerate(chatFiles, 1):
            try:
                # 如果base64数据包含前缀，需要去掉
                if isinstance(image_base64, str) and image_base64.startswith(('data:image', 'image')):
                    image_base64 = image_base64.split(',')[1]
                
                # 构造ollama API请求
                ollama_url = "http://localhost:11434/api/generate"
                ollama_payload = {
                    "model": "llava:latest",
                    "prompt": "这是企业微信的截图，图片中出现了报错信息，你需要提取报错信息",
                    "images": [image_base64],
                    "stream": False
                }
                
                # 发送请求到ollama
                ollama_response = requests.post(
                    ollama_url, 
                    json=ollama_payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                if ollama_response.status_code == 200:
                    response_data = ollama_response.json()
                    recognized_text = response_data.get('response', '未能识别图片内容')
                    
                    # 强制清理文本格式
                    if recognized_text:
                        # 移除所有特殊字符，只保留基本的中英文和标点
                        import re
                        recognized_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', recognized_text)
                        recognized_text = re.sub(r'\s+', ' ', recognized_text)  # 合并多个空白字符
                        recognized_text = recognized_text.strip()
                    
                    image_descriptions.append(f"图片{index}：{recognized_text}")
                    logger.info(f'图片{index}识别成功: {recognized_text[:50]}...')  # 只记录前50个字符
                else:
                    error_msg = f"图片{index}：识别失败(状态码:{ollama_response.status_code})"
                    image_descriptions.append(error_msg)
                    logger.error(error_msg)
                    
            except Exception as e:
                error_msg = f"图片{index}：识别过程出现错误({str(e)})"
                image_descriptions.append(error_msg)
                logger.error(error_msg)
        
        # 组合所有图片的描述
        result = f"\n\n【图片信息】用户一共上传了{total_images}个图片"
        if image_descriptions:
            # 使用普通的换行符而不是过多的换行
            result += "\n" + "\n".join(image_descriptions)
        
        # 最终清理
        import re
        result = re.sub(r'\n{3,}', '\n\n', result)  # 最多保留两个连续换行
        
        logger.info(f'最终返回结果: {result[:100]}...')  # 记录结果的前100个字符
        print("result:", result)
        return result
        
    except Exception as e:
        error_msg = f"图片识别过程出现错误: {str(e)}"
        logger.error(error_msg)
        return error_msg

    
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
        model = "qwen2.5:14b"
        messages = data.get('messages', [])
        stream = data.get('stream', False)
        temperature = data.get('temperature', 1.0)
        api_key = data.get('apiKey', '')
        isRAG = data.get('isRAG', False)
        sim = data.get('similarity', 0.5)
        is_summary = data.get('isSummary', False)
        isapp = data.get('isapp', False)

        # 更新响应处理器配置并重置human_content
        response_handler.update(isRAG, model, sim)
        response_handler.human_content = None  # 重置human_content
        
        # 验证必要参数
        if not messages:
            return jsonify({"error": "Messages are required"}), 400
        
        # 解析用户消息（从 messages 数组中提取第一条用户消息）
        user_message = ""
        chatCode = ""
        
        for msg in messages:
            if msg.get('role') == 'user':
                user_message = msg.get('content', '')
                chatCode = msg.get('chatCode', "")
                contentType = msg.get('contentType', "")
                chatFiles = msg.get('chatFiles', "")
                break
        
        if not user_message:
            return jsonify({"error": "No user message found"}), 400
        
        if(contentType == "2" or contentType == "3"):
            chatFiles_descriptions = process_image_content(chatFiles, logger)
            user_message += "\n\n"
            user_message += chatFiles_descriptions

        logger.info(f'用户消息: {user_message}')

        
        # 可选：验证 API Key
        # if api_key and not api_key.startswith('sk-'):
        #     logger.warning(f'无效的 API Key: {api_key}')
            # 可以选择是否拒绝请求
        
        if stream:
            # 流式响应 - 真正的流式输出
            def generate_stream():
                try:
                    if is_summary:
                        response_text = response_handler.get_history_summary()
                        # 检查是否需要转人工
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
                    else:
                        # 真正的流式输出实现
                        from queue import Queue
                        import threading
                        
                        # 使用队列在线程间传递数据
                        queue = Queue()
                        
                        def run_async_stream():
                            """在新线程中运行异步流式处理"""
                            async def async_stream():
                                try:
                                    async for chunk in response_handler.run_stream(user_message):
                                        if chunk:  # 只发送非空chunk
                                            queue.put(('data', chunk))
                                except Exception as e:
                                    queue.put(('error', str(e)))
                                finally:
                                    queue.put(('done', None))
                            
                            # 运行异步函数
                            asyncio.run(async_stream())
                        
                        # 启动异步流式处理线程
                        thread = threading.Thread(target=run_async_stream)
                        thread.daemon = True  # 设置为守护线程
                        thread.start()
                        
                        # 从队列中读取数据并发送
                        while True:
                            try:
                                msg_type, data = queue.get(timeout=60)  # 60秒超时
                                
                                if msg_type == 'done':
                                    break
                                elif msg_type == 'error':
                                    logger.error(f'流式处理错误: {data}')
                                    error_chunk = {
                                        "id": f"chatcmpl-{int(time.time()*1000)}",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model,
                                        "chatCode": chatCode,
                                        "choices": [{
                                            "delta": {
                                                "content": f"\n错误: {data}"
                                            },
                                            "index": 0,
                                            "finish_reason": "error"
                                        }]
                                    }
                                    yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                                    break
                                elif msg_type == 'data':
                                    # 发送正常的响应块
                                    chunk_data = {
                                        "id": f"chatcmpl-{int(time.time()*1000)}",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model,
                                        "chatCode": chatCode,
                                        "choices": [{
                                            "delta": {
                                                "content": data
                                            },
                                            "index": 0,
                                            "finish_reason": None
                                        }]
                                    }
                                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                                    
                            except Exception as e:
                                logger.error(f'队列获取超时: {e}')
                                break
                        
                        # 等待线程结束
                        thread.join(timeout=5)
                        
                        # 检查是否需要转人工（在流式处理完成后）
                        human = 0 if response_handler.human_content is None else 1
                        
                        # 发送结束标记
                        end_chunk = {
                            "id": f"chatcmpl-{int(time.time()*1000)}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "chatCode": chatCode,
                            "human": human,
                            "choices": [{
                                "delta": {},
                                "index": 0,
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
                
                except Exception as e:
                    logger.error(f'流式输出错误: {str(e)}')
                    error_chunk = {
                        "id": f"chatcmpl-{int(time.time()*1000)}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "chatCode": chatCode,
                        "choices": [{
                            "delta": {
                                "content": f"\n严重错误: {str(e)}"
                            },
                            "index": 0,
                            "finish_reason": "error"
                        }]
                    }
                    yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                
                # 发送最终结束标记
                yield "data: [DONE]\n\n"
            
            return Response(stream_with_context(generate_stream()), mimetype='text/event-stream')
        
        else:
            # 非流式响应
            if is_summary:
                response_text = response_handler.get_history_summary()
            else:
                response_text = generate_ai_response(user_message)

            # 检查是否需要转人工
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
    app.run(
        host='0.0.0.0', 
        port=19006, 
        debug=FLASK_ENV == 'development',
        threaded=True
    )
    # http://192.10.220.191:19006/v1/chat/completions