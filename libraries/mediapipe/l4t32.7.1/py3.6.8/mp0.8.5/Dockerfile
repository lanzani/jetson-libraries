FROM federicolanzani/opencv-cuda-jetson:l4t-base-r32.7.1-python3.6.9

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    nano \
    git \
    curl \
    unzip \
    libhdf5-serial-dev  \
    hdf5-tools  \
    libhdf5-dev  \
    zlib1g-dev  \
    zip \
    libjpeg8-dev  \
    liblapack-dev  \
    libblas-dev  \
    gfortran \
    python3-pip \
    libopencv-core-dev  \
    libopencv-highgui-dev  \
    libopencv-calib3d-dev  \
    libopencv-features2d-dev  \
    libopencv-imgproc-dev  \
    libopencv-video-dev

RUN pip3 install -U pip testresources setuptools==49.6.0

RUN pip3 install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig

RUN git clone https://github.com/PINTO0309/mediapipe-bin \
    && cd mediapipe-bin \
    && ./v0.8.5/download.sh

RUN cd mediapipe-bin \
    && unzip v0.8.5.zip -d v0.8.5 \
    && pip3 install v0.8.5/v0.8.5/numpy119x/py36/mediapipe-0.8.5_cuda102-cp36-cp36m-linux_aarch64.whl \
    && pip3 install dataclasses
