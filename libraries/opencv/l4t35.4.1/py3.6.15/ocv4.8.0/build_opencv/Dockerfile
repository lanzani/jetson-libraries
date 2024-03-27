#FROM ghcr.io/lanzani/python:l4t35.4.1-py3.6.15
FROM dustynv/python:r35.4.1

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Image used to build Opencv 4.8.0, pakcages .sh installer available at /tmp/build_opencv/opencv/build, Build details in build_opencv/build_opencv.sh"
LABEL org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive
#ENV PYTHONPATH=/usr/local/lib/python3.6/site-packages

# Install python 3.6
# Install tzdata
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt -y install tzdata

RUN apt install -y software-properties-common

# Install dependencies
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.6-full && \
    apt clean

RUN python3.6 -m ensurepip --upgrade


RUN apt update && apt upgrade -y && \
    apt install -y \
    nano \
    build-essential \
    cmake \
    git \
    gfortran \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libavresample-dev \
    libcanberra-gtk3-module \
    libdc1394-22-dev \
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
    python-dev \
    python-numpy \
    python3-dev \
    python3-numpy \
    python3-matplotlib \
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
    pkg-config \
    python-dev \
    python-numpy \
    python3-dev \
    python3-numpy \
    libeigen3-dev \
    python3.6-dev \
    python3-pip

RUN pip3 install --upgrade pip &&  \
    pip3 uninstall -y numpy  matplotlib && \
    pip3 install wheel numpy && \
    pip3 install matplotlib

RUN mkdir "build_opencv"

COPY build_opencv.sh build_opencv/

RUN cd build_opencv && ./build_opencv.sh

# Create package
RUN cd /tmp/build_opencv/opencv/build \
    && make package
