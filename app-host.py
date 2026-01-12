import os
import uuid
import logging

from flask import Flask, send_file, jsonify, request, abort, render_template
from werkzeug.utils import secure_filename
from datetime import datetime
from faster_whisper import WhisperModel
from dotenv import load_dotenv

ENV_PATH = os.environ.get('ENV_PATH') if os.environ.get('ENV_PATH') != None else '.env'

load_dotenv(ENV_PATH)

logging.basicConfig(level=logging.DEBUG if os.getenv('LOG_LEVEL') == 'DEBUG' else logging.INFO)

logging.info(f'ENV_PATH={ENV_PATH}')

WHISPER_MODEL = 'medium' if os.environ.get('WHISPER_MODEL') != None else os.environ.get('WHISPER_MODEL')

def seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


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
        beam_size=5,
        chunk_length=30,
        vad_filter=True
    )
    
    DIR = os.path.dirname(input_path)
    FILE_NAME = os.path.splitext(os.path.basename(input_path))[0]
    TXT_FILE_PATH = f"{os.path.join(DIR, f'{FILE_NAME}.txt')}"
    
    with open(TXT_FILE_PATH, "w", encoding="utf-8") as f:
        for seg in segments:
            line = f"[{seconds_to_hms(seg.start)} → {seconds_to_hms(seg.end)}]: {seg.text}\n"
            f.write(line)
            logging.info(line.strip())
    
    return f'{FILE_NAME}.txt'


app = Flask(__name__)

# Конфигурация для загрузки файлов
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB
app.config['ALLOWED_EXTENSIONS'] = {'webm', 'mkv', 'mp4', 'jpg', 'mp3', 'wav', 'm4a'}

# Создаем папку для загрузок, если ее нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.before_request
def check_file_size():
    if request.content_length and request.content_length > app.config['MAX_CONTENT_LENGTH']:
        abort(413, description="File too large")


def get_unique_filename(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4()}.{ext}"
    return unique_name


def get_date_based_folder():
    """Генерация пути папки на основе текущей даты"""
    now = datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    
    date_folder = os.path.join(year, month, day)
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], date_folder)
    
    os.makedirs(full_path, exist_ok=True)
    
    return full_path, date_folder


def allowed_file(filename):
    """Проверка разрешенных расширений файлов"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Главная страница с интерфейсом"""
    return render_template('index.html')


@app.route('/health_check', methods=['GET'])
def health_check():
    return jsonify({"status": "success"})


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Эндпоинт для загрузки файла"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_folder, date_folder = get_date_based_folder()
        
        uniq_name = get_unique_filename(filename)
        file_path = os.path.join(upload_folder, uniq_name)
        file.save(file_path)
        
        # Получаем информацию о файле
        file_size = os.path.getsize(file_path)
        
        return jsonify({
            "message": "File uploaded successfully",
            "original_filename": filename,
            "unique_filename": uniq_name,
            "date_folder": date_folder,
            "file_path": file_path,
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2)
        }), 200
    
    return jsonify({"error": "File type not allowed"}), 400


@app.route('/api/run', methods=['POST'])
def run_transcription():
    """Запуск транскрибации"""
    data = request.json
    if not data or 'unique_filename' not in data or 'date_folder' not in data:
        return jsonify({"error": "Missing parameters"}), 400
    
    unique_filename = data['unique_filename']
    date_folder = data['date_folder']
    
    # Разбираем путь даты на компоненты
    year, month, day = date_folder.split('/')
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], year, month, day, unique_filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    try:
        # Запускаем транскрибацию
        txt_filename = transcribe_large_audio(file_path, model_size=WHISPER_MODEL or "medium")
        
        return jsonify({
            "message": "Transcription completed",
            "txt_filename": txt_filename,
            "date_folder": date_folder,
            "download_url": f"/api/download/{year}/{month}/{day}/{txt_filename}"
        }), 200
    
    except Exception as e:
        logging.error(f"Transcription error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/download/<year>/<month>/<day>/<name>', methods=['GET'])
def download_file(year, month, day, name):
    """Скачать файл по имени"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], year, month, day, name)
        
        if not os.path.exists(file_path):
            abort(404, description="File not found")
        
        if not os.path.isfile(file_path):
            abort(400, description="Not a file")
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=name
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/status/<year>/<month>/<day>/<name>', methods=['GET'])
def check_file_exists(year, month, day, name):
    """Проверить существование файла"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], year, month, day, name)
        exists = os.path.exists(file_path)
        
        return jsonify({
            "exists": exists,
            "file_path": file_path
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Создаем папку для шаблонов, если ее нет
    os.makedirs('templates', exist_ok=True)
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )