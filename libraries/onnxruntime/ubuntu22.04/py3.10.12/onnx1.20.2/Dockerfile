FROM nvcr.io/nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
# TO ghcr.io/lanzani/onnxruntime:ubuntu22.04-cu12.6-py3.10.12-onnx1.20.2-trt10.3.0

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries \
      org.opencontainers.image.description="onnxruntime-gpu" \
      org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
        python3-pip  \
        libnvinfer10  \
        libnvinfer-plugin10 \
        libnvonnxparsers10 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip3 install --no-cache onnxruntime-gpu==1.20.2 tensorrt==10.3.0
