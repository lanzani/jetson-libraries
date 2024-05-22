FROM dustynv/tensorflow2:r35.4.1

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Tensorflow 2.11.0 + Opencv 4.8.0 built with cuda support on jetson nano usign python 3.8.10."
LABEL org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/usr/local/lib/python3.8/site-packages

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
    python3.8 \
    python3.8-dev \
    && apt-get clean

RUN python3.8 -m pip install pybind11 \
    && python3.8 -m pip install Cython==0.29.36 \
    && python3.8 -m pip install setuptools==49.6.0

# RUN python3.8 -m pip install numpy==1.19.4

COPY OpenCV-4.8.0-aarch64.sh ./tmp

RUN cd /tmp && ./OpenCV-4.8.0-aarch64.sh --prefix=/usr/local --skip-license --exclude-subdir

ENV PYTHONPATH "/usr/local/lib/python3.8/site-packages:${PYTHONPATH}"
