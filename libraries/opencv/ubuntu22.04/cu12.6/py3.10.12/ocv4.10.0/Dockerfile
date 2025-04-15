FROM nvcr.io/nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
# TO ghcr.io/lanzani/opencv:ubuntu22.04-cu12.6-py3.10.12-ocv4.10.0

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Opencv 4.10.0 built with cuda support on jetson nano usign python 3.10.12. Build details in build_opencv/build_opencv.sh"
LABEL org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
        python3 \
        python3-numpy \
        libtesseract4 \
        libatlas3-base \
        libswscale5 \
        libavformat58 \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        libgtk-3-0 \
        libcanberra-gtk3-module \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
    && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY OpenCV-unknown-x86_64.sh /tmp/

RUN cd /tmp && \
    chmod +x OpenCV-unknown-x86_64.sh && \
    ./OpenCV-unknown-x86_64.sh --prefix=/usr/local --skip-license --exclude-subdir && \
    rm -rf /tmp/*
