FROM ghcr.io/lanzani/python:l4t32.7.1-py3.10.11

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Opencv 4.8.0 built with cuda support on jetson nano usign python 3.10.11. Build details in build_opencv/build_opencv.sh"
LABEL org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y --autoremove

RUN apt-get update && apt-get install -y \
    libtesseract4 \
    libatlas3-base \
    && apt-get clean

RUN pip3 install numpy

COPY OpenCV-4.8.0-aarch64.sh ./tmp

RUN cd /tmp && ./OpenCV-4.8.0-aarch64.sh --prefix=/usr/local --skip-license --exclude-subdir
