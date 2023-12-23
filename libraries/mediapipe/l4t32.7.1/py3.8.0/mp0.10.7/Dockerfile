FROM ghcr.io/lanzani/opencv:l4t32.7.1-py3.8.0-ocv4.8.0

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Mediapipe 0.10.7 with opencv 4.8.0 built with cuda, for python 3.8.0."
LABEL org.opencontainers.image.licenses=MIT

RUN apt-get update \
    && apt-get upgrade

RUN apt-get install -y \
    libopencv-core-dev \
    libopencv-highgui-dev \
    libopencv-calib3d-dev \
    libopencv-features2d-dev \
    libopencv-imgproc-dev  \
    libopencv-video-dev

COPY mediapipe-0.10.7-cp38-cp38-linux_aarch64.whl .

RUN python3.8 -m pip install --upgrade pip

RUN python3.8 -m pip install mediapipe-0.10.7-cp38-cp38-linux_aarch64.whl \
    && python3.8 -m pip install matplotlib

