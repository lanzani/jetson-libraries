#FROM nvcr.io/nvidia/l4t-base:35.4.1
FROM dustynv/python:r35.4.1

LABEL org.opencontainers.image.source=https://github.com/lanzani/jetson-libraries
LABEL org.opencontainers.image.description="Python 3.6.15 for jetson l4t35.4.1"
LABEL org.opencontainers.image.licenses=MIT

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y --autoremove && apt clean

# Install tzdata
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt -y install tzdata

RUN apt install -y software-properties-common

# Install dependencies
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.6-full && \
    apt clean

RUN python3.6 -m ensurepip --upgrade
