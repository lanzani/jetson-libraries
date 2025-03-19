FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Opencv 4.10.0 built with cuda support on jetson nano usign python 3.10.12. Build details in build_opencv/build_opencv.sh"
LABEL org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y --autoremove

# Install tzdata
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

# Install nvidia codec sdk
COPY build_opencv/libnvcuvid.so /usr/local/cuda/lib64/
COPY build_opencv/libnvidia-encode.so /usr/local/cuda/lib64/
COPY build_opencv/cuviddec.h /usr/local/cuda/include/
COPY build_opencv/nvcuvid.h /usr/local/cuda/include/
COPY build_opencv/nvEncodeAPI.h /usr/local/cuda/include/

# Install runtime dependencies
RUN apt install -y \
    libtesseract4 \
    libatlas3-base

COPY OpenCV-unknown-aarch64.sh ./tmp

RUN cd /tmp && ./OpenCV-unknown-aarch64.sh --prefix=/usr/local --skip-license --exclude-subdir
