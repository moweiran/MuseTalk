export HTTP_PROXY=http://127.0.0.1:20171
export HTTPS_PROXY=http://127.0.0.1:20171

端口号

sudo ufw status
sudo ufw allow 5001
sudo ufw reload

1. 后台运行 api 接口服务
   nohup python fast_api.py > fastapi.log 2>&1 &
2. 查看 log
   tail -f fastapi.log
3. 查找进程 ID
   ps aux | grep fast_api.py
4. Kill it using the process ID
   kill -9 <process_id>
