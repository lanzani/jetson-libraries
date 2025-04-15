FROM ghcr.io/lanzani/ml:l4t36.4.0-cu12.6-py3.10.12-ocv4.10.0-onnx1.20.2-trt10.3.0
# TO ghcr.io/lanzani/ml:l4t36.4.0-cu12.6-py3.10.12-ocv4.10.0-onnx1.20.2-trt10.3.0-pt2.6.0

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries \
      org.opencontainers.image.description="onnxruntime-gpu build for jetson, source: https://pypi.jetson-ai-lab.dev/jp6/cu126/onnxruntime-gpu/1.20.2" \
      org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y python3-pip && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN wget https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/6cc/6ecfe8a5994fd/torch-2.6.0-cp310-cp310-linux_aarch64.whl#sha256=6cc6ecfe8a5994fd6d58fb6d6eb73ff2437428bb4953f3ebaa409f83a5f4db99 && \
    wget https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/aa2/2da8dcf4c4c8d/torchvision-0.21.0-cp310-cp310-linux_aarch64.whl#sha256=aa22da8dcf4c4c8dc897e7922b1ef25cb0fe350e1a358168be87a854ad114531 && \
    pip3 install --no-cache torch-2.6.0-cp310-cp310-linux_aarch64.whl torchvision-0.21.0-cp310-cp310-linux_aarch64.whl && \
    rm torch-2.6.0-cp310-cp310-linux_aarch64.whl torchvision-0.21.0-cp310-cp310-linux_aarch64.whl
