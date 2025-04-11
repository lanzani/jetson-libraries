FROM ghcr.io/lanzani/opencv:ubuntu22.04-cu12.6-py3.10.12-ocv4.10.0

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

RUN python3 -m pip install onnxruntime-gpu==1.20.2 tensorrt==10.3.0