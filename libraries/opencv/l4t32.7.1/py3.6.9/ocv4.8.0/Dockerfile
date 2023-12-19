FROM  nvcr.io/nvidia/l4t-base:r32.7.1

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Opencv 4.8.0 built with cuda support on jetson nano usign python 3.6.9. Build details in build_opencv/build_opencv.sh"
LABEL org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/usr/local/lib/python3.6/site-packages

RUN apt-get update && apt-get upgrade -y --autoremove

RUN apt-get update && apt-get -y install \
    python3-pip \
    libtesseract4 \
    libatlas3-base \
    python3-numpy \
    && apt-get clean


COPY OpenCV-4.8.0-aarch64.sh ./tmp

RUN cd /tmp && ./OpenCV-4.8.0-aarch64.sh --prefix=/usr/local --skip-license --exclude-subdir

# RUN echo opencv_version
# RUN python3 -c "import cv2; print('OpenCV version:', str(cv2.__version__)); print(cv2.getBuildInformation())"
