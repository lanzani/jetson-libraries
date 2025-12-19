FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
# TO ghcr.io/lanzani/opencv:ubuntu22.04-cu12.8-py3.10.12-ocv4.10.0-build

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Image used to build Opencv 4.10.0, pakcages .sh installer available at /tmp/build_opencv/opencv/build, Build details in build_opencv/build_opencv.sh"
LABEL org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y --autoremove
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    nano \
    build-essential \
    cmake \
    git \
    gfortran \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libcanberra-gtk3-module \
    libeigen3-dev \
    libglew-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer1.0-dev \
    libgtk-3-dev \
    libjpeg-dev \
    libjpeg8-dev \
    libjpeg-turbo8-dev \
    liblapack-dev \
    liblapacke-dev \
    libopenblas-dev \
    libpng-dev \
    libpostproc-dev \
    libswscale-dev \
    libtbb-dev \
    libtbb2 \
    libtesseract-dev \
    libtiff-dev \
    libv4l-dev \
    libxine2-dev \
    libxvidcore-dev \
    libx264-dev \
    pkg-config \
    qv4l2 \
    v4l-utils \
    zlib1g-dev \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    libxine2-dev \
    libglew-dev \
    libtiff5-dev \
    zlib1g-dev \
    libjpeg-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libpostproc-dev \
    libswscale-dev \
    libeigen3-dev \
    libtbb-dev \
    libgtk2.0-dev \
    libeigen3-dev

# Install nvidia codec sdk
COPY ./Video_Codec_SDK_12.1.14/Video_Codec_SDK_12.1.14/Lib/linux/stubs/x86_64/*.so* /usr/local/cuda/lib64/stubs/
COPY ./Video_Codec_SDK_12.1.14/Video_Codec_SDK_12.1.14/Interface/*.h /usr/local/cuda/include/
RUN echo "/usr/local/cuda/lib64/stubs\n" > /etc/ld.so.conf.d/900_video-extractor.conf && ldconfig

# Build gst-plugins-bad to have nvcodec plugin
RUN apt update && apt install wget meson libgstreamer-plugins-base1.0-dev pkg-config libgstreamer1.0-dev cmake -y && \
    wget https://gstreamer.freedesktop.org/src/gst-plugins-bad/gst-plugins-bad-1.20.3.tar.xz && \
    tar -xf gst-plugins-bad-1.20.3.tar.xz && \
    cd gst-plugins-bad-1.20.3 && \
    meson setup build \
    -Dgpl=enabled \
    -Dnvcodec=enabled \
    -Dtests=disabled \
    -Dexamples=disabled && \
    ninja -C build && \
    ninja -C build install

COPY build_opencv.sh ./

RUN chmod +x build_opencv.sh

RUN ./build_opencv.sh

RUN cd workspace/opencv-4.10.0/release \
    && make package
