FROM dustynv/l4t-pytorch:r36.2.0

# === Tensorflow =======================================================================================================

# install prerequisites - https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html#prereqs
RUN apt update && apt upgrade -y && \
    apt install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

RUN pip3 install -U testresources setuptools==65.5.0

# install Python TF dependencies
RUN pip3 install -U numpy==1.22 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig packaging h5py==3.7.0

# TensorFlow
RUN pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60 tensorflow==2.15.0+nv24.05

# === OpenCV ===========================================================================================================

# Install nvidia codec sdk
COPY libnvcuvid.so /usr/local/cuda/lib64/
COPY libnvidia-encode.so /usr/local/cuda/lib64/
COPY cuviddec.h /usr/local/cuda/include/
COPY nvcuvid.h /usr/local/cuda/include/
COPY nvEncodeAPI.h /usr/local/cuda/include/

# Install runtime dependencies
RUN apt update && apt install -y \
    libtesseract4 \
    libatlas3-base

COPY OpenCV-unknown-aarch64.sh ./tmp

RUN cd /tmp && ./OpenCV-unknown-aarch64.sh --prefix=/usr/local --skip-license --exclude-subdir

# === Ultralytics ======================================================================================================
RUN pip3 install ultralytics
