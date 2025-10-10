from fastapi import FastAPI, HTTPException
import torch
import uvicorn
from player import Player

app = FastAPI()

player = Player()  # Global player instance

@app.get("/gpu_check")
def gpu_check():
    """
    测试GPU是否可用
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        if gpu_id >= gpu_count:
            print(f"Warning: GPU {gpu_id} not available, using GPU 0 instead")
            gpu_id = 0
        else:
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        print("CUDA is not available, using CPU")
    return {"status": "success", "message": "GPU is available"}

@app.get("/inference")
def hello():
    """
    测试播放
    """
    player.inference("rtmps://rtmp.icommu.cn/live/livestream")
    return {"status": "success", "message": "Streaming started"}

@app.get("/play")
def test_play():
    """
    测试播放
    """
    player.play("rtmps://rtmp.icommu.cn/live/livestream")
    return {"status": "success", "message": "Streaming started"}

@app.get("/stop")
def test_stop():
    """Stop streaming"""
    player.stop()
    return {"status": "success", "message": "Streaming stopped"}

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5002
    )