FROM python:3.8

WORKDIR /usr/src/app

COPY peekingduck/training/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
