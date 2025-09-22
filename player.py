

import subprocess
import threading
import time
class Player:
    def __init__(self):
        self.mp4_thread = None # 用于存储线程对象
        self.stop_event = threading.Event()  # 用于控制线程停止
    
    def start_streaming(self, rtmp_url:str):
        """
        开始播放MP4文件到RTMP服务器
        """
        stream_cmd = [
            'ffmpeg',
            '-re',
            '-stream_loop', '-1',
            '-r', '30',
            '-i', f"./audio_0.mp4",
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-profile:v', 'baseline',
            '-level', '3.1',
            '-pix_fmt', 'yuv420p',
            '-g', '300',
            '-keyint_min', '60',
            '-b:v', '1200k',
            '-maxrate', '1200k',
            '-bufsize', '1800k',
            '-c:a', 'aac',
            '-ar', '16000',
            '-ac', '1',
            '-b:a', '64k',
            '-shortest',
            '-f', 'flv',
            '-flvflags', 'no_duration_filesize',
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
            print("Streaming is already running. No new thread will be started.")
            return

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
        