FROM python:3.7

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
COPY . ${APP_HOME}
WORKDIR ${APP_HOME}
# Build PeekingDuck locally (only for testing PeekingDuck server)
RUN pip install . --no-deps

EXPOSE 5000