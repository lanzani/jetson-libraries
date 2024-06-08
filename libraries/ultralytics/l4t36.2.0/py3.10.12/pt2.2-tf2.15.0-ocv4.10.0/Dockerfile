FROM ghcr.io/lanzani/opencv:l4t36.2.0-py3.10.12-ocv4.10.0

# === Pytorch ==========================================================================================================
RUN apt update && apt upgrade -y \
    && apt install -y python3-pip libopenblas-base libopenmpi-dev libomp-dev

RUN pip3 install 'Cython<3'
RUN pip3 install numpy

COPY torch-2.3.0-cp310-cp310-linux_aarch64.whl /.
COPY torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl /.
COPY torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl /.

RUN pip3 install torch-2.3.0-cp310-cp310-linux_aarch64.whl torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl

RUN rm torch-2.3.0-cp310-cp310-linux_aarch64.whl torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl

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

## === Ultralytics =====================================================================================================
RUN pip3 install ultralytics
