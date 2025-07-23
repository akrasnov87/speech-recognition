#!/bin/bash

ffmpeg -i $1 -f mp3 -ab 192000 -vn ./data/audio.mp3
ffmpeg -i ./data/audio.mp3 ./data/audio.wav