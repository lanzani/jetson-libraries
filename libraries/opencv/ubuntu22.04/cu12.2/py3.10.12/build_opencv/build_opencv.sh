#!/bin/bash
#
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#

version="4.10.0"
folder="workspace"

set -e

echo "** Remove other OpenCV first"
apt -y purge *libopencv*

echo "------------------------------------"
echo "** Install requirement (1/4)"
echo "------------------------------------"
apt update
apt install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
apt install -y python3.10-dev python3-dev python3-numpy
apt install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev
apt install -y libv4l-dev v4l-utils qv4l2
apt install -y curl unzip

apt install -y gfortran libeigen3-dev libblas-dev libblas64-dev libatlas-base-dev liblapack-dev libopenblas-dev libgsl-dev


echo "------------------------------------"
echo "** Download opencv "${version}" (2/4)"
echo "------------------------------------"
mkdir $folder
cd ${folder}
curl -L https://github.com/opencv/opencv/archive/${version}.zip -o opencv-${version}.zip
curl -L https://github.com/opencv/opencv_contrib/archive/${version}.zip -o opencv_contrib-${version}.zip
unzip opencv-${version}.zip
unzip opencv_contrib-${version}.zip
rm opencv-${version}.zip opencv_contrib-${version}.zip
cd opencv-${version}/


# compute capabilities: https://developer.nvidia.com/cuda-gpus

echo "------------------------------------"
echo "** Build opencv "${version}" (3/4)"
echo "------------------------------------"
mkdir release
cd release/
cmake -D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D CUDA_ARCH_BIN="6.1;8.9" \
-D CUDA_ARCH_PTX="" \
-D CUDA_FAST_MATH=ON \
-D CUDNN_VERSION='8.9' \
-D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
-D OPENCV_DNN_CUDA=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${version}/modules \
-D WITH_CUBLAS=ON \
-D WITH_CUDA=ON \
-D WITH_GSTREAMER=ON \
-D WITH_LIBV4L=ON \
-D WITH_OPENGL=ON \
-D WITH_LAPACK=ON \
-D BUILD_opencv_python3=ON \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D BUILD_EXAMPLES=OFF \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local .. \
-D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs
make -j$(nproc)


echo "------------------------------------"
echo "** Install opencv "${version}" (4/4)"
echo "------------------------------------"
make install
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PYTHONPATH=/usr/local/lib/python3.10/site-packages/:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc


echo "** Install opencv "${version}" successfully"
echo "** Bye :)"