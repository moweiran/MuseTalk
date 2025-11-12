1. 后台运行 api 接口服务
   nohup python fast_api.py > fastapi.log 2>&1 &
2. 查看 log
   tail -f fastapi.log
3. 查找进程 ID
   ps aux | grep fast_api.py
4. Kill it using the process ID
   kill -9 <process_id>
