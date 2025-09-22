

import subprocess
import threading
import time
class Player:
    def __init__(self):
        self.mp4_thread = None # 用于存储线程对象
        self.stop_event = threading.Event()  # 用于控制线程停止
        self.current_rtmp_url = None
    
    def start_streaming(self, rtmp_url:str):
        """
        开始播放MP4文件到RTMP服务器
        """
        stream_cmd = [
            'ffmpeg',
            '-re',
            '-stream_loop', '-1',
            '-r', '30',
            '-i', f"./data/video/mofei.mp4",
            '-preset', 'medium',
            '-profile:v', 'baseline',
            '-level', '3.1',
            '-pix_fmt', 'yuv420p',
            '-g', '300',
            '-b:v', '1200k',
            '-maxrate', '1200k',
            '-bufsize', '1800k',
            '-s', '720x1280',
            '-ar', '16000',
            '-ac', '1',
            '-b:a', '64k',
            '-f', 'flv',
            '-flvflags',
            '-flags +low_delay',
            rtmp_url,
        ]
        process = subprocess.Popen(stream_cmd)
        try:
            while not self.stop_event.is_set():
                time.sleep(1)  # 检查停止事件
        finally:
            print("Stopping process...")
            process.terminate()  # 尝试优雅停止
            try:
                process.wait(timeout=5)  # 等待进程退出
            except subprocess.TimeoutExpired:
                print("Process did not terminate, killing it...")
                process.kill()  # 强制停止
            print("Process stopped.")
            
    def play(self, rtmp_url):
        # 检查是否已经有线程在运行
        if self.mp4_thread and self.mp4_thread.is_alive():
            if self.current_rtmp_url == rtmp_url:
                print("Streaming is already running with the same URL. No new thread will be started.")
                return
            else:
                print("RTMP URL has changed. Stopping current stream...")
                self.stop()  # 停止当前线程

        # 更新当前的 rtmp_url
        self.current_rtmp_url = rtmp_url

        # 如果没有线程在运行，则启动新的线程
        self.stop_event.clear()  # 确保停止事件未设置
        self.mp4_thread = threading.Thread(target=self.start_streaming, args=(rtmp_url,))
        self.mp4_thread.start()
        print("Streaming started.")
        
    def stop(self):
        # 停止线程
        if self.mp4_thread and self.mp4_thread.is_alive():
            self.stop_event.set()  # 通知线程停止
            self.mp4_thread.join()  # 等待线程退出
            print("Streaming stopped.")
        else:
            print("No streaming thread is running.")
        self.current_rtmp_url = None  # 清空当前的 rtmp_url
            
    def inference(self, rtmp_url:str):
         # 启动播放
        self.play(rtmp_url)

        # 模拟运行 10 秒后停止
        time.sleep(10)
        self.stop()

        # 再次尝试启动播放
        self.play(rtmp_url)
        

if __name__ == "__main__":
    pass
        