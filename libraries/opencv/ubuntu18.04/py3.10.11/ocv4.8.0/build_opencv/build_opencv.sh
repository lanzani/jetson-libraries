#!/usr/bin/env bash
# 2019 Michael de Gans, https://github.com/mdegans/nano_build_opencv/blob/master/build_opencv.sh


set -e

# change default constants here:
readonly PREFIX=/usr/local  # install prefix, (can be ~/.local for a user install)
readonly DEFAULT_VERSION=4.8.0  # controls the default version (gets reset by the first argument)
readonly CPUS=$(nproc)  # controls the number of jobs

# better board detection. if it has 6 or more cpus, it probably has a ton of ram too
if [[ $CPUS -gt 5 ]]; then
    # something with a ton of ram
    JOBS=$CPUS
else
    JOBS=1  # you can set this to 4 if you have a swap file
    # otherwise a Nano will choke towards the end of the build
fi

cleanup () {
# https://stackoverflow.com/questions/226703/how-do-i-prompt-for-yes-no-cancel-input-in-a-linux-shell-script
    while true ; do
        echo "Do you wish to remove temporary build files in /tmp/build_opencv ? "
        if ! [[ "$1" -eq "--test-warning" ]] ; then
            echo "(Doing so may make running tests on the build later impossible)"
        fi
        read -p "Y/N " yn
        case ${yn} in
            [Yy]* ) rm -rf /tmp/build_opencv ; break;;
            [Nn]* ) exit ;;
            * ) echo "Please answer yes or no." ;;
        esac
    done
}

setup () {
    cd /tmp
    if [[ -d "build_opencv" ]] ; then
        echo "It appears an existing build exists in /tmp/build_opencv"
        cleanup
    fi
    mkdir build_opencv
    cd build_opencv
}

git_source () {
    echo "Getting version '$1' of OpenCV"
    git clone --depth 1 --branch "$1" https://github.com/opencv/opencv.git
    git clone --depth 1 --branch "$1" https://github.com/opencv/opencv_contrib.git
}

install_dependencies () {
    # open-cv has a lot of dependencies, but most can be found in the default
    # package repository or should already be installed (eg. CUDA).
    echo "Installing build dependencies."
    apt-get update --fix-missing
#    apt-get dist-upgrade -y --autoremove

#    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
#    sudo dpkg -i cuda-keyring*.deb
#    sudo apt-get update
#    sudo apt-get install -y cuda
#
#    apt-get install -y libcudnn8 --allow-change-held-packages
#    apt-get install -y libcudnn8-dev

}

configure () {

    # Change CUDNN_VERSION with your gpu capability: https://developer.nvidia.com/cuda-gpus

#    -D PYTHON_DEFAULT_EXECUTABLE=`which python3.10`

    local CMAKEFLAGS="
        -D PYTHON3_INCLUDE_DIR=/usr/include/python3.10
        -D PYTHON3_INCLUDE_PATH=/usr/include/python3.10
        -D PYTHON3_EXECUTABLE=`which python3.10`
        -D CMAKE_BUILD_TYPE=RELEASE
        -D CMAKE_INSTALL_PREFIX=${PREFIX}
        -D WITH_CUDA=ON
        -D OPENCV_EXTRA_MODULES_PATH=/tmp/build_opencv/opencv_contrib/modules
        -D EIGEN_INCLUDE_PATH=/usr/include/eigen3
        -D CUDNN_VERSION='8.5'
        -D CUDA_FAST_MATH=ON
        -D OPENCV_DNN_CUDA=ON
        -D BUILD_EXAMPLES=OFF
        -D BUILD_opencv_python2=ON
        -D BUILD_opencv_python3=ON
        -D WITH_GSTREAMER=ON
        -D WITH_LIBV4L=ON
        -D WITH_OPENGL=ON
        -D WITH_CUDNN=ON
        -D OPENCV_GENERATE_PKGCONFIG=ON
        -D OPENCV_ENABLE_NONFREE=ON
        -D WITH_CUBLAS=ON
        "

    if [[ "$1" != "test" ]] ; then
        CMAKEFLAGS="
        ${CMAKEFLAGS}
        -D BUILD_PERF_TESTS=OFF
        -D BUILD_TESTS=OFF"
    fi

    echo "cmake flags: ${CMAKEFLAGS}"

    cd opencv
    mkdir build
    cd build
    cmake ${CMAKEFLAGS} .. 2>&1 | tee -a configure.log
}

main () {

    local VER=${DEFAULT_VERSION}

    # parse arguments
    if [[ "$#" -gt 0 ]] ; then
        VER="$1"  # override the version
    fi

    if [[ "$#" -gt 1 ]] && [[ "$2" == "test" ]] ; then
        DO_TEST=1
    fi

    # prepare for the build:
    setup
    install_dependencies
    git_source ${VER}

    if [[ ${DO_TEST} ]] ; then
        configure test
    else
        configure
    fi

    # start the build
    make -j${JOBS} 2>&1 | tee -a build.log

    if [[ ${DO_TEST} ]] ; then
        make test 2>&1 | tee -a test.log
    fi

    # avoid a sudo make install (and root owned files in ~) if $PREFIX is writable
    if [[ -w ${PREFIX} ]] ; then
        make install 2>&1 | tee -a install.log
    else
        make install 2>&1 | tee -a install.log
    fi

    ldconfig

#    cleanup --test-warning

}

main "$@"
