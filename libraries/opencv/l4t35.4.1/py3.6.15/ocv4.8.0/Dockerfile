FROM ghcr.io/lanzani/python:l4t35.4.1-py3.6.15

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Opencv 4.8.0 built with cuda support on jetson nano usign python 3.6.15. Build details in build_opencv/build_opencv.sh"
LABEL org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y --autoremove

RUN apt-get update && apt-get install -y \
    libtesseract4 \
    libatlas3-base \
    && apt-get clean

RUN python3.6 -m pip install numpy --upgrade

COPY OpenCV-4.8.0-aarch64.sh ./tmp

RUN cd /tmp && ./OpenCV-4.8.0-aarch64.sh --prefix=/usr/local --skip-license --exclude-subdir

ENV PYTHONPATH "/usr/local/lib/python3.6/site-packages:${PYTHONPATH}"
