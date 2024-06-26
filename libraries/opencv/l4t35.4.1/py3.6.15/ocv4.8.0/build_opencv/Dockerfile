FROM ghcr.io/lanzani/python:l4t35.4.1-py3.6.15

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Image used to build Opencv 4.8.0, pakcages .sh installer available at /tmp/build_opencv/opencv/build, Build details in build_opencv/build_opencv.sh"
LABEL org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive
#ENV PYTHONPATH=/usr/local/lib/python3.6/site-packages

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
    pip3 uninstall -y numpy  matplotlib

RUN python3.6 -m pip install --upgrade pip &&  \
    python3.6 -m pip install numpy

RUN mkdir "build_opencv" && pwd

COPY build_opencv.sh build_opencv/

RUN cd build_opencv && ./build_opencv.sh

# Create package
RUN cd /tmp/build_opencv/opencv/build \
    && make package

ENV PYTHONPATH "/usr/local/lib/python3.6/site-packages:${PYTHONPATH}"
