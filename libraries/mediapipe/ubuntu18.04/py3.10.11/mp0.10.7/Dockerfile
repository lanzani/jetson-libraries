FROM ghcr.io/lanzani/opencv:ubuntu18.04-py3.10.11-ocv4.8.0

RUN apt-get update \
    && apt-get upgrade

RUN apt-get install -y \
    libopencv-core-dev \
    libopencv-highgui-dev \
    libopencv-calib3d-dev \
    libopencv-features2d-dev \
    libopencv-imgproc-dev  \
    libopencv-video-dev

RUN python3.10 -m pip install --upgrade pip

RUN python3.10 -m pip install mediapipe==0.10.7 && \
    python3.10 -m pip install matplotlib
