FROM ghcr.io/lanzani/opencv:l4t36.4.0-cu12.6-py3.10.12-ocv4.10.0
# TO ghcr.io/lanzani/ml:l4t36.4.0-cu12.6-py3.10.12-ocv4.10.0-onnx1.20.2-trt10.3.0

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

RUN wget https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/f6e/2baa664069470/onnxruntime_gpu-1.20.2-cp310-cp310-linux_aarch64.whl#sha256=f6e2baa664069470c6574219a79aba315e26c76db49d347678a5a273f1c41c9a && \
    pip3 install --no-cache onnxruntime_gpu-1.20.2-cp310-cp310-linux_aarch64.whl "numpy<2.0.0" && \
    rm onnxruntime_gpu-1.20.2-cp310-cp310-linux_aarch64.whl
