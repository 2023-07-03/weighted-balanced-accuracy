FROM amd64/python:3.6-slim-buster

RUN apt-get update
RUN apt-get install -y  vim

ADD requirements.txt /server/requirements.txt
RUN pip3 install -r /server/requirements.txt


ADD URLNet /server/URLNet

WORKDIR /server
