#FROM federicolanzani/ubuntu18.04-cuda:python3.10.11
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu18.04



#ENV PYTHONPATH=/usr/lib/python3.10/site-packages

# === PYTHON 3.10.11 ===================================================================================================

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
    libncursesw5-dev \
    locales \
    libgpm2

# Build python
RUN apt-get install --reinstall ca-certificates -y \
    && git clone https://github.com/JetsonHacksNano/build_python.git \
    && cd build_python \
    && sed -i 's/^sudo //' build_python3.sh \
    && bash ./build_python3.sh --version 3.10; exit 0

# Make local repository
RUN cd build_python  \
    && sed -i 's/^sudo //' make_apt_repository.sh \
    && bash ./make_apt_repository.sh --version 3.10

# Install python
RUN apt-get install -y python3.10-full \
    python3-testresources \
    && python3.10 -m ensurepip --upgrade

# === OPENCV ===========================================================================================================

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
    python3.10-dev \
    libhdf5-serial-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 uninstall -y numpy matplotlib \
    && pip3 install --no-cache numpy matplotlib tables

RUN mkdir "build_opencv"

COPY build_opencv.sh build_opencv/

RUN cd build_opencv && ./build_opencv.sh

# Create package
RUN cd /tmp/build_opencv/opencv/build \
    && make package