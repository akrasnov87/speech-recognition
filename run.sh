#!/bin/bash

python3 -m venv speech
source speech/bin/activate

echo "run $SCRIPT_NAME.py"

if [ -n "$VIDEO_FILE" ]; then
    ffmpeg -i $VIDEO_FILE -f mp3 -ab 192000 -vn /data/audio.mp3
    ffmpeg -i /data/audio.mp3 $WAV_FILE_PATH
fi

if [ -n "$SCRIPT_NAME" ]; then
    python3 /app/$SCRIPT_NAME.py
else
    python3 /app/app-fasted.py
fi