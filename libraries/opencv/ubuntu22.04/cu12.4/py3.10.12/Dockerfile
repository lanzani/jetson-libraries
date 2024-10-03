FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Opencv 4.10.0 built with cuda support on jetson nano usign python 3.10.12. Build details in build_opencv/build_opencv.sh"
LABEL org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y --autoremove

# Install tzdata
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

# Install runtime dependencies
RUN apt update && \
    apt install -y \
    nano \
    libtesseract4 \
    libatlas3-base \
    python3 \
    python3-numpy \
    libswscale5 \
    libavformat58 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgtk-3-0 \
    libcanberra-gtk3-module \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav

COPY OpenCV-unknown-x86_64.sh ./tmp

RUN cd /tmp && ./OpenCV-unknown-x86_64.sh --prefix=/usr/local --skip-license --exclude-subdir

RUN rm -rf /tmp/*

RUN echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc && \
    echo 'export PYTHONPATH=/usr/local/lib/python3.10/site-packages/:$PYTHONPATH' >> ~/.bashrc
