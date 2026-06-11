# InfiniteTalk 拉取式 worker —— 便携镜像
# 钉死的是「实测能跑通生成」的版本组合：torch2.5.1 / cu124 / py3.12 / flash-attn 2.7.4
#   + numpy2.0.2 / scipy1.13.1 / numba0.60 / llvmlite0.43。
# 适配 Hopper / Ampere（H100 / H200 / H800 / A100 / A800）。
# Blackwell（B200 / RTX PRO 6000）需要 cu128 / torch2.7，另出 tag，不在此镜像。
#
# 权重(~90G)不打进镜像 —— 运行时下载到挂载卷 /app/weights（首次下，之后复用）。
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com

# Python 3.12（匹配 flash-attn cp312 预编译轮子）+ ffmpeg(ffprobe) + git
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git ffmpeg && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch 2.5.1 + cu124（实测组合；Actions 在境外构建，用官方源没问题）
RUN python -m pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124
# flash-attn 预编译轮子（cp312 / cu12torch2.5，免现场编译）
RUN python -m pip install \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
# xformers（对应 torch2.5 / cu124）
RUN python -m pip install xformers==0.0.28.post3 \
        --index-url https://download.pytorch.org/whl/cu124

# 仓库依赖
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r requirements.txt
# 钉死实测能生成的科学栈（覆盖 requirements 里的 numpy<2）：
# numba0.60 会拉 numpy2.0.2，这套就是跑通 60 段窗口的版本（之后那次失败是 RAM OOM，与此无关）。
RUN python -m pip install numpy==2.0.2 scipy==1.13.1 numba==0.60.0 llvmlite==0.43.0
# 权重下载 + worker 运行依赖
RUN python -m pip install "huggingface_hub[cli]" modelscope hf_transfer httpx

# 代码（.dockerignore 已排除 weights/ _worker/ uploads/ 等）
COPY . /app
RUN chmod +x docker/entrypoint.sh docker/download_weights.sh

ENTRYPOINT ["/app/docker/entrypoint.sh"]
