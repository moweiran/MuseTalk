CUDA 12.8的显卡可能会有错误，请自行解决

错误一：
OR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
triton 2.0.0 requires cmake, which is not installed.
triton 2.0.0 requires lit, which is not installed.


```
conda install cmake
conda install lit

```

错误二
mim install mmengine
Traceback (most recent call last):
  File "/home/richsos/miniconda3/envs/MuseTalk/bin/mim", line 7, in <module>
    sys.exit(cli())
  File "/home/richsos/miniconda3/envs/MuseTalk/lib/python3.10/site-packages/click/core.py", line 1442, in __call__
    return self.main(*args, **kwargs)
  File "/home/richsos/miniconda3/envs/MuseTalk/lib/python3.10/site-packages/click/core.py", line 1363, in main
    rv = self.invoke(ctx)
  File "/home/richsos/miniconda3/envs/MuseTalk/lib/python3.10/site-packages/click/core.py", line 1830, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
  File "/home/richsos/miniconda3/envs/MuseTalk/lib/python3.10/site-packages/click/core.py", line 1226, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/home/richsos/miniconda3/envs/MuseTalk/lib/python3.10/site-packages/click/core.py", line 794, in invoke
    return callback(*args, **kwargs)
  File "/home/richsos/miniconda3/envs/MuseTalk/lib/python3.10/site-packages/mim/commands/install.py", line 72, in cli
    exit_code = install(list(args), index_url=index_url, is_yes=is_yes)
  File "/home/richsos/miniconda3/envs/MuseTalk/lib/python3.10/site-packages/mim/commands/install.py", line 128, in install
    install_args += ['-f', get_mmcv_full_find_link(mmcv_base_url)]
  File "/home/richsos/miniconda3/envs/MuseTalk/lib/python3.10/site-packages/mim/commands/install.py", line 165, in get_mmcv_full_find_link
    torch_v, cuda_v = get_torch_cuda_version()
  File "/home/richsos/miniconda3/envs/MuseTalk/lib/python3.10/site-packages/mim/utils/utils.py", line 340, in get_torch_cuda_version
    raise err
  File "/home/richsos/miniconda3/envs/MuseTalk/lib/python3.10/site-packages/mim/utils/utils.py", line 338, in get_torch_cuda_version
    import torch
  File "/home/richsos/miniconda3/envs/MuseTalk/lib/python3.10/site-packages/torch/__init__.py", line 229, in <module>
    from torch._C import *  # noqa: F403
ImportError: /home/richsos/miniconda3/envs/MuseTalk/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```

conda create -n MuseTalk python==3.10
conda activate MuseTalk

conda install cudatoolkit=11.7

# Option 1: Using pip
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Option 2: Using conda
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia


# 激活环境后检查
python -c "import torch; print(torch.version.cuda)"

pip install -r requirements.txt


pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"

楼上的，MMCV 2.0.1不支持cu128，你改成 mim install mmvc==2.1.0 就好了

cuda 目前12.8，但12.8的显卡不支持，只能使用11.8
conda install cudatoolkit=11.7


conda deactivate  # 先退出当前环境（如果不是在base环境中）
conda deactivate
conda remove -n MuseTalk --all

conda env remove -n MuseTalk

conda env list
```

通过 RPM Fusion 仓库安装（推荐）
这是比较推荐的方法，因为 RPM Fusion 提供了为 Enterprise Linux（包括 CentOS Stream）预编译的 FFmpeg 包，安装和管理都比较方便。

```
sudo dnf update -y
sudo dnf install -y epel-release

sudo dnf install -y https://download1.rpmfusion.org/free/el/rpmfusion-free-release-9.noarch.rpm

sudo dnf install -y https://download1.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-9.noarch.rpm

sudo dnf install -y ffmpeg ffmpeg-devel
ffmpeg -version
上面方案不行
用conda安装
conda install -c conda-forge ffmpeg
```


sudo vim /etc/hosts

121.15.167.250 digitalhuman.richinfo.cn

sudo killall -HUP mDNSResponder