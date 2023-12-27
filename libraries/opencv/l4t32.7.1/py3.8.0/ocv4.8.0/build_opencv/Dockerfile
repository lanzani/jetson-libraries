FROM  nvcr.io/nvidia/l4t-base:r32.7.1

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Image used to build Opencv 4.8.0, pakcages .sh installer available at /tmp/build_opencv/opencv/build, Build details in build_opencv/build_opencv.sh"
LABEL org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/usr/local/lib/python3.8/site-packages

RUN apt-get update && apt-get upgrade -y --autoremove

# Install tzdata
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

# Install dependencies
RUN apt-get update && apt-get upgrade && apt-get -y install \
    git \
    git-buildpackage \
    python3.8 \
    debhelper \
    autoconf \
    libncursesw5-dev

# Install build dependencies TODO check duplicates
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
    python-pip \
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
    python3.8 \
    python3.8-dev

RUN mkdir "build_opencv"

COPY build_opencv.sh build_opencv/

RUN apt-get remove -y python3-numpy python3-matplotlib \
    && apt-get install -y python3-pip \
    && python3.8 -m pip install pybind11 \
    && python3.8 -m pip install Cython==0.29.36 \
    && python3.8 -m pip install setuptools==49.6.0

RUN python3.8 -m pip install numpy==1.19.4

RUN cd build_opencv && ./build_opencv.sh

# Create package
RUN cd /tmp/build_opencv/opencv/build \
    && make package
