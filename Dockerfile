FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
ARG MODE=whisper
ARG SCRIPT_MODE=app-fasted.py

RUN apt update && apt upgrade -y
RUN apt install --no-install-recommends python3 python3-pip python3-venv ffmpeg -y

RUN pip3 install --upgrade pip

COPY ./requirements-${MODE}.txt /app/requirements.txt
COPY ./${MODE}.sh /app/run.sh
RUN chmod 777 /app/run.sh
COPY ./app-${MODE}.py /app/app-${MODE}.py

WORKDIR /app

RUN pip3 install -r requirements.txt

COPY ./datasets/ /data/
RUN ffmpeg -i /data/video.mp4 /data/audio.wav
COPY ./.env.${MODE} /app/.env
RUN python3 /app/app-${MODE}.py
RUN rm /app/.env
RUN rm -rf /data

RUN apt clean
RUN rm -rf /var/lib/apt/lists/*

VOLUME /data

RUN sed -i 's/\r//g' /app/run.sh

ENTRYPOINT ["/bin/bash"] 

CMD ["/app/run.sh"]