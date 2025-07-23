FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN apt update && apt upgrade -y
RUN apt install --no-install-recommends python3 python3-pip python3-venv ffmpeg -y

RUN pip3 install --upgrade pip

#COPY ./requirements.txt /app/requirements.txt
COPY ./speech /app/speech
COPY ./run.sh /app/run.sh
RUN chmod 777 /app/run.sh
COPY ./app.py /app/app.py
COPY ./app-fasted.py /app/app-fasted.py

WORKDIR /app

#RUN pip3 install -r requirements.txt

RUN apt clean
RUN rm -rf /var/lib/apt/lists/*

VOLUME /data

RUN sed -i 's/\r//g' /app/run.sh

ENTRYPOINT ["/bin/bash", "/app/run.sh"] 