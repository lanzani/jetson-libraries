FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
#FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Image used to build Opencv 4.10.0, pakcages .sh installer available at /tmp/build_opencv/opencv/build, Build details in build_opencv/build_opencv.sh"
LABEL org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y --autoremove

RUN apt update && apt upgrade -y && \
    apt install -y \
    nano \
    build-essential \
    cmake \
    git \
    gfortran \
    libatlas-base-dev \
    libcanberra-gtk3-module \
    libeigen3-dev \
    libglew-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer1.0-dev \
    libjpeg8-dev \
    libjpeg-turbo8-dev \
    liblapack-dev \
    liblapacke-dev \
    libopenblas-dev \
    libpostproc-dev \
    libtbb-dev \
    libtbb2 \
    libtesseract-dev \
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
    libtbb2 \
    libtbb-dev \
    libpq-dev \
    libxine2-dev \
    libglew-dev \
    libtiff5-dev \
    zlib1g-dev \
    libavutil-dev \
    libpostproc-dev \
    libeigen3-dev \
    libtbb-dev \
    pkg-config \
    libeigen3-dev

# Install nvidia codec sdk
COPY libnvcuvid.so /usr/local/cuda/lib64/
COPY libnvidia-encode.so /usr/local/cuda/lib64/
COPY cuviddec.h /usr/local/cuda/include/
COPY nvcuvid.h /usr/local/cuda/include/
COPY nvEncodeAPI.h /usr/local/cuda/include/

RUN apt update && \
    apt install -y build-essential cmake gcc g++ git libgtk-3-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev && \
    apt install -y libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev &&\
    apt install -y python3 python3-dev python3-numpy &&\
    apt install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libopenexr-dev libwebp-dev &&\
    apt install -y libv4l-dev v4l-utils qv4l2 &&\
    apt install -y curl unzip &&\
    apt install -y gfortran libeigen3-dev libblas-dev libblas64-dev libatlas-base-dev liblapack-dev libopenblas-dev libgsl-dev


RUN wget https://developer.download.nvidia.com/compute/cudnn/9.1.1/local_installers/cudnn-local-repo-ubuntu2204-9.1.1_1.0-1_amd64.deb && \
    dpkg -i cudnn-local-repo-ubuntu2204-9.1.1_1.0-1_amd64.deb && \
    cp /var/cudnn-local-repo-ubuntu2204-9.1.1/cudnn-*-keyring.gpg /usr/share/keyrings/

RUN apt update && apt upgrade -y && \
    apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libnccl-dev libnccl2 --allow-change-held-packages && \
    apt install -y cudnn

COPY build_opencv.sh ./

RUN chmod +x build_opencv.sh

RUN ./build_opencv.sh


