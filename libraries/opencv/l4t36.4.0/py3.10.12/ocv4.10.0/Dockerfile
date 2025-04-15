FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0
# TO ghcr.io/lanzani/opencv:l4t36.4.0-cu12.6-py3.10.12-ocv4.10.0

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries \
      org.opencontainers.image.description="Opencv 4.10.0 built with cuda support on jetson nano usign python 3.10.12. Build details in build_opencv/build_opencv.sh" \
      org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
        libtesseract4 \
        libatlas3-base && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install nvidia codec sdk
COPY build_opencv/libnvcuvid.so /usr/local/cuda/lib64/
COPY build_opencv/libnvidia-encode.so /usr/local/cuda/lib64/
COPY build_opencv/cuviddec.h /usr/local/cuda/include/
COPY build_opencv/nvcuvid.h /usr/local/cuda/include/
COPY build_opencv/nvEncodeAPI.h /usr/local/cuda/include/

COPY OpenCV-unknown-aarch64.sh /tmp/

RUN cd /tmp && \
    chmod +x OpenCV-unknown-aarch64.sh && \
    ./OpenCV-unknown-aarch64.sh --prefix=/usr/local --skip-license --exclude-subdir && \
    rm -rf /tmp/*
