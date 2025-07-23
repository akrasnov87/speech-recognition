# Описание

Утилита для генерации текста из аудиодорожки. Основана на https://github.com/openai/whisper

## Установка

<pre>
sudo apt update && sudo apt install ffmpeg
</pre>

<pre>
python3 -m venv speech
source speech/bin/activate
pip install -r requirements.txt
</pre>

Создание `requirements.txt` из локльных пакетов:

`pip freeze > requirements.txt`

## Экспорт mp3 из mp4

<pre>
ffmpeg -i video.mp4 -f mp3 -ab 192000 -vn music.mp3
ffmpeg -i music.mp3 audio.wav
</pre>

Может появиться сообщение о том, что вам не хватает некоторых кодеков для MP4.

В этом случае просто запустите: `aptitude search codecname`

Оригинал статьи: https://askubuntu.com/questions/174287/how-do-i-convert-an-mp4-to-an-mp3

## Hugging Face

`huggingface-cli login`

Далее вводим токен для авторизации

## Docker

### Сборка
`docker build -t akrasnov87/audio2text:0.0.1 .`

### Использование
<pre>
docker run -it --rm --env-file ./.env --name audio2text -v ./data:/data:rw akrasnov87/audio2text:0.0.1
</pre>

### Переменные

* SCRIPT_NAME: str - скрипт запуска `app` или `app-fasted`;
* LOG_LEVEL: str - логирование `DEBUG` или `INFO`;
* VIDEO_FILE: str - путь к видеофайлу, можно не указывать;
* WAV_FILE_PATH: str - путь к аудио файлу;
* WHISPER_MODEL: str - модель, по умолчанию `medium`;
* CHUNK_LENGTH: int - размер блока аудио для разбиения, по умолчанию `300000 мс`;
* HF_TOKEN: str - токен авторизации на `huggingface`.

Пример:

<pre>
SCRIPT_NAME='app-fasted'
LOG_LEVEL='INFO'
VIDEO_FILE='data/video.webm'
WAV_FILE_PATH='data/audio.wav'
WHISPER_MODEL='medium'
CHUNK_LENGTH=300000
HF_TOKEN='**********'
</pre>