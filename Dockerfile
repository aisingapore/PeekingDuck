# https://cloud.google.com/ai-platform/training/docs/using-containers
# If your training configuration uses NVIDIA A100 GPUs, then your container must use CUDA 11 or later

FROM gcr.io/deeplearning-platform-release/base-cu110

ARG APP_HOME="/Repositories"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y \
    curl \
    ffmpeg \
    rsync \
    software-properties-common \
    ssh \
    unzip \
    vim \
    python3-pip \
    wget && \
    apt-get clean

COPY requirements.txt /tmp/
WORKDIR /tmp/
RUN pip install -r requirements.txt

# COPY <repo_dir> <target location within Docker container>
COPY . ${APP_HOME}/PeekingDuck
WORKDIR ${APP_HOME}/PeekingDuck

