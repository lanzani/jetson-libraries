FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
# TO ghcr.io/lanzani/opencv:ubuntu22.04-cu12.8-py3.10.12-ocv4.10.0

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Opencv 4.10.0 built with cuda support on jetson nano usign python 3.10.12. Build details in build_opencv/build_opencv.sh"
LABEL org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
    nano \
    python3 \
    python3-numpy \
    python3-pip \
    libtesseract4 \
    libatlas3-base \
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
    gstreamer1.0-libav \
    && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install nvidia codec sdk
COPY ./build_opencv/Video_Codec_SDK_12.1.14/Video_Codec_SDK_12.1.14/Lib/linux/stubs/x86_64/*.so* /usr/local/cuda/lib64/stubs/
COPY ./build_opencv/Video_Codec_SDK_12.1.14/Video_Codec_SDK_12.1.14/Interface/*.h /usr/local/cuda/include/
RUN echo "/usr/local/cuda/lib64/stubs\n" > /etc/ld.so.conf.d/900_video-extractor.conf && ldconfig

# Build gst-plugins-bad to have nvcodec plugin
RUN apt update && apt install wget meson libgstreamer-plugins-base1.0-dev pkg-config libgstreamer1.0-dev cmake -y && \
    wget https://gstreamer.freedesktop.org/src/gst-plugins-bad/gst-plugins-bad-1.20.3.tar.xz && \
    tar -xf gst-plugins-bad-1.20.3.tar.xz && \
    cd gst-plugins-bad-1.20.3 && \
    meson setup build \
    -Dgpl=enabled \
    -Dnvcodec=enabled \
    -Dtests=disabled \
    -Dexamples=disabled && \
    ninja -C build && \
    ninja -C build install

COPY OpenCV-unknown-x86_64.sh /tmp/

RUN cd /tmp && \
    chmod +x OpenCV-unknown-x86_64.sh && \
    ./OpenCV-unknown-x86_64.sh --prefix=/usr/local --skip-license --exclude-subdir && \
    rm -rf /tmp/*

RUN pip cache purge
