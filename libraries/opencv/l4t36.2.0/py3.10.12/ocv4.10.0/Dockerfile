FROM nvcr.io/nvidia/l4t-jetpack:r36.2.0

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Opencv 4.10.0 built with cuda support on jetson nano usign python 3.10.12. Build details in build_opencv/build_opencv.sh"
LABEL org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y --autoremove

# Install tzdata
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

# Install dependencies
RUN apt-get update && apt-get upgrade \
    && apt-get remove -y python3-numpy python3-matplotlib \
    && apt-get install -y  \
    libtesseract4 \
    libatlas3-base \
    python3-pip \
    && apt-get clean

RUN pip install numpy

COPY OpenCV-4.10.0-aarch64.sh ./tmp

RUN cd /tmp && ./OpenCV-4.10.0-aarch64.sh --prefix=/usr/local --skip-license --exclude-subdir
