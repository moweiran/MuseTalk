from fastapi import FastAPI, HTTPException
import uvicorn
from player import Player

app = FastAPI()

player = Player()  # Global player instance


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