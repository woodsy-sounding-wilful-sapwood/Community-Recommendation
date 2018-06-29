FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev
COPY . /rec_api
WORKDIR /rec_api
RUN pip3 install -r requirements.txt
ENTRYPOINT python3 -m flask run --host 0.0.0.0 --port 3445
