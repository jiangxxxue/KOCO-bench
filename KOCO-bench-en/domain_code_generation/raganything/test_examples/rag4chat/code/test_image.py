import base64
import requests
from PIL import Image
import io
import os

def image_to_base64(image_input, max_size=(1024, 1024)):
    """
    将图片转换为base64编码
    
    Args:
        image_input: 图片路径或URL
        max_size: 最大尺寸限制，避免图片过大
    
    Returns:
        str: base64编码的图片数据
    """
    
    try:
        # 判断是本地文件还是URL
        if image_input.startswith(('http://', 'https://')):
            # 处理网络图片
            print(f"正在下载图片: {image_input}")
            response = requests.get(image_input)
            response.raise_for_status()
            
            # 获取图片格式
            content_type = response.headers.get('content-type', '')
            if 'jpeg' in content_type or 'jpg' in content_type:
                image_format = 'JPEG'
            elif 'png' in content_type:
                image_format = 'PNG'
            elif 'gif' in content_type:
                image_format = 'GIF'
            else:
                image_format = 'JPEG'  # 默认格式
            
            image_data = response.content
            print(f"图片下载成功，大小: {len(image_data)} 字节")
            
        else:
            # 处理本地文件
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"文件不存在: {image_input}")
            
            print(f"正在读取本地图片: {image_input}")
            with open(image_input, 'rb') as f:
                image_data = f.read()
            
            # 根据文件扩展名确定格式
            ext = os.path.splitext(image_input)[1].lower()
            if ext in ['.jpg', '.jpeg']:
                image_format = 'JPEG'
            elif ext == '.png':
                image_format = 'PNG'
            elif ext == '.gif':
                image_format = 'GIF'
            else:
                image_format = 'JPEG'
            
            print(f"本地图片读取成功，大小: {len(image_data)} 字节")
        
        # 使用PIL处理图片并调整大小
        image = Image.open(io.BytesIO(image_data))
        print(f"原始图片尺寸: {image.size}, 格式: {image.format}")
        
        # 调整图片大小（如果超过限制）
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            print(f"调整后图片尺寸: {image.size}")
        
        # 转换为RGB格式（如果需要）
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        
        # 转换为base64
        buffer = io.BytesIO()
        image.save(buffer, format=image_format, quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print(f"Base64编码长度: {len(img_str)}")
        return img_str
        
    except Exception as e:
        print(f"转换失败: {str(e)}")
        return None

def test_image_conversion():
    """测试图片转换功能"""
    
    # 测试用例
    test_cases = [
        # 你可以在这里添加实际的测试图片URL或路径
        "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png",
        # "/path/to/your/local/image.jpg",  # 本地图片路径
    ]
    
    print("=== 图片转Base64测试 ===\n")
    
    for i, image_input in enumerate(test_cases, 1):
        print(f"测试 {i}: {image_input}")
        print("-" * 50)
        
        base64_result = image_to_base64(image_input)
        
        if base64_result:
            print(f"✅ 转换成功!")
            print(f"📦 Base64前50个字符: {base64_result[:50]}...")
            print(f"📏 Base64总长度: {len(base64_result)}")
            
            # 验证base64是否有效
            try:
                # 尝试解码验证
                decoded = base64.b64decode(base64_result)
                print(f"🔍 解码验证成功，解码后长度: {len(decoded)}")
            except Exception as e:
                print(f"❌ Base64解码验证失败: {e}")
        else:
            print("❌ 转换失败")
        
        print("\n" + "="*60 + "\n")

# 交互式测试函数
def interactive_test():
    """交互式测试"""
    print("=== 交互式图片转Base64工具 ===")
    print("请输入图片URL或本地文件路径（输入 'quit' 退出）:")
    
    while True:
        user_input = input("\n图片地址: ").strip()
        
        if user_input.lower() == 'quit':
            print("退出程序")
            break
            
        if not user_input:
            print("请输入有效的图片地址")
            continue
            
        print(f"\n正在处理: {user_input}")
        base64_result = image_to_base64(user_input)
        
        if base64_result:
            print(f"\n✅ 转换成功!")
            print(f"📦 Base64编码长度: {len(base64_result)}")
            print(f"📋 Base64数据 (前100字符):")
            print(base64_result)
            print(base64_result[:100] + "..." if len(base64_result) > 100 else base64_result)
            
            # 询问是否保存到文件
            save_option = input("\n是否保存到文件? (y/n): ").strip().lower()
            if save_option == 'y':
                filename = input("请输入文件名 (默认: image_base64.txt): ").strip()
                if not filename:
                    filename = "image_base64.txt"
                
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(base64_result)
                    print(f"✅ 已保存到 {filename}")
                except Exception as e:
                    print(f"❌ 保存失败: {e}")
        else:
            print("❌ 转换失败，请检查图片地址是否正确")

if __name__ == "__main__":
    # 运行测试
    # test_image_conversion()  # 批量测试
    
    # 或运行交互式测试
    interactive_test()