FROM ghcr.io/lanzani/opencv:l4t32.7.1-py3.8.0-ocv4.8.0

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Mediapipe 0.10.7 with opencv 4.8.0 built with cuda, for python 3.8.0."
LABEL org.opencontainers.image.licenses=MIT

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y  \
    mesa-common-dev  \
    libegl1-mesa-dev  \
    libgles2-mesa-dev  \
    mesa-utils  \
    pkg-config  \
    zip  \
    g++  \
    zlib1g-dev  \
    unzip  \
    python3-dev  \
    python3.8-dev  \
    software-properties-common \
    cmake \
    curl

# TODO check deps
RUN apt-get install -y \
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

RUN apt install -y \
    libopencv-core-dev \
    libopencv-highgui-dev \
    libopencv-calib3d-dev \
    libopencv-features2d-dev \
    libopencv-imgproc-dev  \
    libopencv-video-dev

RUN add-apt-repository ppa:webupd8team/java \
    && apt-get update \
    && apt install -y openjdk-11-jdk

RUN python3.8 -m pip install pip setuptools --upgrade

# Install bazel
RUN wget https://github.com/bazelbuild/bazel/releases/download/6.4.0/bazel-6.4.0-linux-arm64 \
    && chmod +x bazel-6.4.0-linux-arm64 \
    && mv bazel-6.4.0-linux-arm64 /usr/local/bin/bazel

# Install protoc
RUN curl -OL https://github.com/google/protobuf/releases/download/v3.19.1/protoc-3.19.1-linux-aarch_64.zip \
    && unzip protoc-3.19.1-linux-aarch_64.zip -d protoc3 \
    && cp -r protoc3/bin/* /usr/local/bin/ \
    && cp -r protoc3/include/* /usr/local/include/ \
    && ldconfig

# Link opencv
RUN ln -s /usr/local/include/opencv4/ /usr/include/aarch64-linux-gnu/ \
    && ln -s /usr/local/include/opencv4/ /usr/include/
#    && ln -s /usr/include/opencv4/opencv2/ /usr/include

# Update gcc and g++
RUN apt-get update \
    && apt-get install -y  \
    gcc-8 \
    g++-8 \
    && cd /usr/bin \
    && rm gcc g++ \
    && ln -s gcc-8 gcc \
    && ln -s g++-8 g++

# Protoc
COPY mediapipe-0.10.7 /mediapipe-0.10.7

RUN apt-get install -y libopencv-contrib-dev

RUN cd mediapipe-0.10.7 \
    && export TF_CUDA_PATHS=/usr/local/cuda-10.2:/usr/lib/aarch64-linux-gnu:/usr/include \
    && export MEDIAPIPE_DISABLE_GPU=0 \
    && python3.8 setup.py gen_protos  \
    && python3.8 setup.py bdist_wheel

RUN python3.8 -m pip install mediapipe-0.10.7/dist/mediapipe-0.10.7-cp38-cp38-linux_aarch64.whl \
    && python3.8 -m pip uninstall -y matplotlib \
    && python3.8 -m pip install matplotlib

