FROM ghcr.io/lanzani/opencv:ubuntu22.04-cu12.8-py3.10.12-ocv4.10.0
# TO ghcr.io/lanzani/ml:ubuntu22.04-cu12.8-py3.10.12-ocv4.10.0-pt2.8.0

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries \
    org.opencontainers.image.description="PyTorch 2.8.0" \
    org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt update && \
    apt upgrade -y && \
    apt install -y python3-pip && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Upgrade pip
RUN pip3 install --upgrade pip

RUN pip3 install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
