FROM ubuntu:latest
RUN apt-get update -y
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y python3-pip python3-dev python3-tk python3
WORKDIR /rec_api
ADD ./requirements.txt /rec_api/requirements.txt
RUN pip3 install -r requirements.txt
ADD . /rec_api
