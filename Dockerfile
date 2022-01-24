# https://cloud.google.com/ai-platform/training/docs/using-containers
# If your training configuration uses NVIDIA A100 GPUs, then your container must use CUDA 11 or later

FROM gcr.io/deeplearning-platform-release/base-cu110

ARG APP_HOME="/Repositories"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y \
    build-essential \
    bzip2 \
    curl \
    gcc \
    ffmpeg \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libzmq3-dev \
    pkg-config \
    rsync \
    software-properties-common \
    ssh \
    unzip \
    vim \
    python3-pip \
    wget && \
    apt-get clean

COPY . ${APP_HOME}/PeekingDuck
WORKDIR ${APP_HOME}/PeekingDuck

RUN pip install -r requirements.txt