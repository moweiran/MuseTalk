```
#!/bin/bash
echo "正在清理GPU内存..."

# 显示清理前的GPU状态
echo "清理前:"
nvidia-smi

# 杀死所有Python进程（谨慎使用）
echo "杀死Python进程..."
pkill -f python

# 重置GPU会话
echo "重置GPU会话..."
sudo nvidia-smi --reset-gpu-sessions

# 清理系统缓存
echo "清理系统缓存..."
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches

# 重启NVIDIA持久化守护进程
echo "重启NVIDIA服务..."
sudo systemctl restart nvidia-persistenced

echo "清理完成，当前GPU状态:"
nvidia-smi
```
