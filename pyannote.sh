#!/bin/bash

if [ -n "$VIDEO_FILE" ]; then
    ffmpeg -i $VIDEO_FILE $WAV_FILE_PATH
fi

if [ -n "$SCRIPT_NAME" ]; then
    python3 /app/$SCRIPT_NAME.py
else
    python3 /app/app-pyannote.py
fi