from pydub import AudioSegment
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
import glob
import os
import logging

ENV_PATH=os.environ.get('ENV_PATH') if os.environ.get('ENV_PATH') != None else '.env'

load_dotenv(ENV_PATH)

logging.basicConfig(level=logging.DEBUG if os.getenv('LOG_LEVEL') == 'DEBUG' else logging.INFO)

logging.info(f'ENV_PATH={ENV_PATH}')

# путь к wav-файлу
WAV_FILE_PATH='data/audio.wav' if os.getenv('WAV_FILE_PATH') == None else os.getenv('WAV_FILE_PATH')
CHUNK_LENGTH=300000 if os.getenv('CHUNK_LENGTH') == None else int(os.getenv('CHUNK_LENGTH'))
HF_TOKEN=os.getenv('HF_TOKEN')
NUM_SPEAKERS=os.getenv('NUM_SPEAKERS')
WHISPER_MODEL='medium' if os.environ.get('WHISPER_MODEL') != None else os.environ.get('WHISPER_MODEL')

if os.path.isfile(WAV_FILE_PATH) == False:
    logging.info(f'Аудиофайл {WAV_FILE_PATH} не найден.')

# 1. разделение большого файла на маленькие отрывки

audio = AudioSegment.from_wav(WAV_FILE_PATH)
chunks = [audio[i:i + CHUNK_LENGTH] for i in range(0, len(audio), CHUNK_LENGTH)]

DIR=os.path.dirname(WAV_FILE_PATH)
FILE_NAME=os.path.splitext(os.path.basename(WAV_FILE_PATH))[0]

for i, chunk in enumerate(chunks):
    CHUNK_FILE_PATH=f"{os.path.join(DIR, f'{FILE_NAME}_{i}.wav')}"
    chunk.export(CHUNK_FILE_PATH, format="wav")
    logging.debug(f'Создан {CHUNK_FILE_PATH}')

# 2. получение спикеров

"""
Получение спикеров из аудиофайла

Параметры:
----------
path_mask: str - маска пути для поиска wav файлов
hf_token: str - токен для авторизации на huggingface
offset_ms: int - Смещение времени в секундах, по умолчанию 300
num_speakers: int - количество спикеров
"""
def get_speakers(path_mask, hf_token, offset_ms=300, num_speakers=None):
    speakers=[]
    # 1. Инициализация моделей
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)

    # 2. Обработка каждого чанка
    for i, chunk_file in enumerate(sorted(glob.glob(path_mask))):
        offset = i * offset_ms
        logging.debug(f'Обработка chunk diarization {chunk_file}...')

        # Диаризация
        diarization = diarization_pipeline(chunk_file, num_speakers=num_speakers)
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = offset + turn.start
            end = offset + turn.end
            
            item = {
                "start": start,
                "end": end,
                "speaker": speaker
            }

            if len(speakers) == 0:
                speakers.append(item)
            else:
                last = speakers[-1]

                if last.get('speaker') == item.get('speaker'):
                    speakers[-1] = {
                        "start": last.get('start'),
                        "end": item.get('end'),
                        "speaker": item.get('speaker')
                    }
                else:
                    speakers.append(item)

    return speakers

PATH_MASK=f"{os.path.join(DIR, f'{FILE_NAME}_*.wav')}"

SECOND_CHUNK_LENGTH=CHUNK_LENGTH / 1000

speakers = get_speakers(PATH_MASK, HF_TOKEN, SECOND_CHUNK_LENGTH, num_speakers=NUM_SPEAKERS)

# 3. распознавание

def seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

TXT_FILE_PATH=f"{os.path.join(DIR, f'{FILE_NAME}.txt')}"

whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

with open(TXT_FILE_PATH, "w", encoding="utf-8") as f:
    # ASR
    segments, _ = whisper_model.transcribe(WAV_FILE_PATH, vad_filter=True, language="ru", beam_size=5, chunk_length=30)
    for segment in segments:
        start = segment.start
        end = segment.end

        speaker = None

        for s in speakers:
            if s.get('start') != None and s.get('start') >= start:
                speaker = s.get('speaker')
                break
        
        if speaker == None:
            speaker = 'unknown'

        line = f"{speaker} [{seconds_to_hms(start)} → {seconds_to_hms(end)}]: {segment.text}\n"
        f.write(line)
        logging.info(line.strip())
        