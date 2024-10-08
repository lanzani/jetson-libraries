FROM ghcr.io/lanzani/opencv:ubuntu22.04-cu12.2-py3.10.12-ocv4.10.0

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Ultralytics libraries on Ubuntu 22.04 with Python 3.10.12, Pytorch 2.4.1, Tensorflow  and OpenCV 4.10.0"
LABEL org.opencontainers.image.licenses=MIT

# Install necessary packages
RUN apt update && apt install -y \
    python3 \
    python3-pip

# Upgrade pip
RUN pip3 install --upgrade pip

# === Tensorflow =======================================================================================================
RUN pip3 install tensorflow[and-cuda]==2.15.1

# === Pytorch ==========================================================================================================
# Install Python packages
RUN pip3 install numpy scipy

# Install PyTorch and torchvision
RUN pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# === Ultralytics =====================================================================================================
RUN pip3 install ultralytics

RUN pip3 install tensorrt==8.6.1

RUN apt update && apt install -y libnvinfer-plugin8 libnvinfer8
