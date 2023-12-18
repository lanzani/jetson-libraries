FROM nvcr.io/nvidia/l4t-base:r32.7.1

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Python 3.8.0 for jetson."
LABEL org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y  \
    python3.8 \
    python3-pip \
    && apt-get clean
