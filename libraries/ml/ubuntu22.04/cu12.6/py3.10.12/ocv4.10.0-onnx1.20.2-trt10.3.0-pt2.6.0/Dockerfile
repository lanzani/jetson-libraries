FROM ghcr.io/lanzani/ml:ubuntu22.04-cu12.6-py3.10.12-ocv4.10.0-onnx1.20.2-trt10.3.0
# TO ghcr.io/lanzani/ml:ubuntu22.04-cu12.6-py3.10.12-ocv4.10.0-onnx1.20.2-trt10.3.0-pt2.6.0

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries \
      org.opencontainers.image.description="onnxruntime-gpu" \
      org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt update && \
    apt upgrade -y && \
    apt install -y python3-pip && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# === Pytorch ==========================================================================================================
# Install PyTorch and torchvision
RUN pip3 install --no-cache torch torchvision --index-url https://download.pytorch.org/whl/cu126
