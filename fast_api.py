from fastapi import FastAPI, HTTPException
import uvicorn
import asyncio
import signal
import sys
from MuseTalk import MuseTalk_RealTime
from scripts.realtime_inference2 import Avatar
from datetime import datetime
import requests
import os
from urllib.parse import urlparse, unquote
import posixpath

musetalker = MuseTalk_RealTime()
app = FastAPI()

# 指定音频文件保存目录
AUDIO_DIR = "data/audio"
# 确保目录存在
os.makedirs(AUDIO_DIR, exist_ok=True)
#  创建Avatar实例
avatar_instance = None

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化Avatar"""
    initialize_avatar()
    

def initialize_avatar():
    """初始化Avatar实例"""
    global avatar_instance
    if avatar_instance is None:
        print("初始化Avatar实例...")
        avatar_instance = Avatar(
            avatar_id="avator_1",
            video_path="data/video/yongen.mp4",
            bbox_shift=0,
            batch_size=20,
            preparation=False)
        print("Avatar实例初始化完成")
    return avatar_instance

    
@app.get("/hello")
def hello():
    musetalker.sayHello()   
    return {"message": "Hello World from FastAPI!"}


def get_filename_from_url(url):
    """
    从URL中提取文件名
    """
    # 解析URL
    parsed_url = urlparse(url)
    # 获取路径部分
    path = parsed_url.path
    # 使用posixpath处理路径分隔符
    filename = posixpath.basename(path)
    # 解码URL编码的字符
    filename = unquote(filename)
    return filename

@app.get("/inference")
def inference(url:str, rtmp_url:str,filename:str = None):
    """
    下载音频文件的API接口
    """
    """
    使用预初始化的Avatar实例进行推理
    """
    global avatar_instance
    try:
        if filename is None:
            filename = get_filename_from_url(url)
            if not filename:
                filename = "downloaded_audio.wav"
                
        # 保存到指定目录
        filepath = os.path.join(AUDIO_DIR, filename)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        print("开始下载")
        # 保存文件
        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                
        print(f"end of download-audio {filepath} at {datetime.now()}")
        # avatar_id = "avator_1"
        # video_path = "data/video/yongen.mp4"
        # avatar = Avatar(
        #     avatar_id="avator_1",
        #     video_path=video_path,
        #     bbox_shift=0,
        #     batch_size=20,
        #     preparation=False)
        # 使用预初始化的Avatar实例
        if avatar_instance is None:
            print("avatar_instance is None 初始化Avatar实例...")
            avatar_instance = initialize_avatar()
            
        print("Inferring using:", filepath)
        
        avatar_instance.inference(
            filepath, 
            "audio_0", 
            25, 
            False, 
            rtmp_url
        )
        print(f"end of hello {datetime.now()}")
        return {"message": f"推理完成! filename: {filename} at {datetime.now()}"}
    except Exception as e:
        print(f"推理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"下载失败: {str(e)}")

@app.get("/download-audio")
def download_audio(url:str, filename:str = None):
    """
    下载音频文件的API接口
    """
    try:
        if filename is None:
            filename = get_filename_from_url(url)
            if not filename:
                filename = "downloaded_audio.wav"
        
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        
        # 保存文件
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                
        print(f"end of download-audio {filename} at {datetime.now()}")
        return {"message": "音频文件下载成功", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下载失败: {str(e)}")
    
    # Graceful shutdown handling
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown handler"""
    print("Shutting down gracefully...")
    # Add any cleanup code here

def signal_handler(sig, frame):
    print('Received interrupt signal. Shutting down...')
    sys.exit(0)

if __name__ == "__main__":
     # Handle interrupt signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    uvicorn.run(
        "fast_api:app", 
        host="0.0.0.0", 
        port=5001,
        reload=False,  # Set to False for production
        workers=1      # Adjust based on your needs
    )
