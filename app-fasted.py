from faster_whisper import WhisperModel
from dotenv import load_dotenv
import logging
import os

ENV_PATH=os.environ.get('ENV_PATH') if os.environ.get('ENV_PATH') != None else '.env'

load_dotenv(ENV_PATH)

logging.basicConfig(level=logging.DEBUG if os.getenv('LOG_LEVEL') == 'DEBUG' else logging.INFO)

logging.info(f'ENV_PATH={ENV_PATH}')

WAV_FILE_PATH='data/audio.wav' if os.getenv('WAV_FILE_PATH') == None else os.getenv('WAV_FILE_PATH')
WHISPER_MODEL='medium' if os.environ.get('WHISPER_MODEL') != None else os.environ.get('WHISPER_MODEL')

DIR=os.path.dirname(WAV_FILE_PATH)
FILE_NAME=os.path.splitext(os.path.basename(WAV_FILE_PATH))[0]
TXT_FILE_PATH=f"{os.path.join(DIR, f'{FILE_NAME}.txt')}"

def seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

"""
Преобразование аудио дорожки в текст

Параметры:
----------
input_path: str - входная строка
model_size: str - название модели
device: str - устройства cpu или gpu
"""
def transcribe_large_audio(
    input_path,
    model_size="medium",
    device="cpu",
    compute_type="int8",
    language="ru",
):
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )
    
    segments, _ = model.transcribe(
            input_path,
            language=language,
            word_timestamps=True,
            beam_size=5,  # улучшает точность
            chunk_length=30,  # размер чанка в секундах (оптимально 20-30)
            vad_filter=True  # фильтрует паузы (экономит память)
        )
        
    with open(TXT_FILE_PATH, "w", encoding="utf-8") as f:
        for seg in segments:
            line = f"[{seconds_to_hms(seg.start)} → {seconds_to_hms(seg.end)}]: {seg.text}\n"
            f.write(line)
            logging.info(line.strip())

transcribe_large_audio(WAV_FILE_PATH, model_size=WHISPER_MODEL)