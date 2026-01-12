import os
import uuid
import logging
import json
from threading import Lock

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


class TranscriptionProgress:
    """Класс для отслеживания прогресса транскрибации"""
    _instances = {}
    _lock = Lock()
    
    @classmethod
    def get_instance(cls, file_id):
        with cls._lock:
            if file_id not in cls._instances:
                cls._instances[file_id] = cls(file_id)
            return cls._instances[file_id]
    
    @classmethod
    def remove_instance(cls, file_id):
        with cls._lock:
            if file_id in cls._instances:
                del cls._instances[file_id]
    
    def __init__(self, file_id):
        self.file_id = file_id
        self.lines = []
        self.status = "pending"  # pending, processing, completed, error
        self.progress = 0
        self.total_segments = 0
        self.processed_segments = 0
        self.error_message = None
        self.start_time = None
        self.end_time = None
    
    def add_line(self, line):
        self.lines.append(line)
        self.processed_segments += 1
        if self.total_segments > 0:
            self.progress = int((self.processed_segments / self.total_segments) * 100)
    
    def to_dict(self):
        return {
            "file_id": self.file_id,
            "status": self.status,
            "progress": self.progress,
            "total_segments": self.total_segments,
            "processed_segments": self.processed_segments,
            "lines": self.lines[-100:],  # Последние 100 строк
            "total_lines": len(self.lines),
            "error_message": self.error_message,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None
        }


def transcribe_large_audio(
    input_path,
    file_id,
    model_size="medium",
    device="cpu",
    compute_type="int8",
    language="ru",
):
    """Транскрибация с сохранением прогресса"""
    progress = TranscriptionProgress.get_instance(file_id)
    progress.status = "processing"
    progress.start_time = datetime.now()
    
    DIR = os.path.dirname(input_path)
    FILE_NAME = os.path.splitext(os.path.basename(input_path))[0]
    TXT_FILE_PATH = f"{os.path.join(DIR, f'{FILE_NAME}.txt')}"
    
    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        
        # Открываем файл для записи
        with open(TXT_FILE_PATH, "w", encoding="utf-8") as f:
            segments, info = model.transcribe(
                input_path,
                language=language,
                word_timestamps=True,
                beam_size=5,
                chunk_length=30,
                vad_filter=True
            )
            
            # Сохраняем общее количество сегментов для прогресса
            progress.total_segments = 0
            # faster-whisper не предоставляет общее количество сегментов заранее
            # Будем считать сегменты по мере их появления
            
            for seg in segments:
                line = f"[{seconds_to_hms(seg.start)} → {seconds_to_hms(seg.end)}]: {seg.text}\n"
                f.write(line)
                f.flush()  # Принудительно записываем в файл
                
                # Сохраняем в прогресс
                progress.add_line(line.strip())
                
                # Логируем на сервере
                logging.info(f"[{file_id}] {line.strip()}")
            
            # После обработки всех сегментов
            progress.total_segments = progress.processed_segments
            progress.status = "completed"
            progress.end_time = datetime.now()
            
        return f'{FILE_NAME}.txt'
    
    except Exception as e:
        progress.status = "error"
        progress.error_message = str(e)
        progress.end_time = datetime.now()
        logging.error(f"Transcription error for {file_id}: {str(e)}")
        raise


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
        
        # Создаем уникальный ID для отслеживания прогресса
        file_id = f"{date_folder}/{uniq_name}"
        
        return jsonify({
            "message": "File uploaded successfully",
            "original_filename": filename,
            "unique_filename": uniq_name,
            "date_folder": date_folder,
            "file_path": file_path,
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "file_id": file_id
        }), 200
    
    return jsonify({"error": "File type not allowed"}), 400


@app.route('/api/run', methods=['POST'])
def run_transcription():
    """Запуск транскрибации"""
    data = request.json
    if not data or 'unique_filename' not in data or 'date_folder' not in data or 'file_id' not in data:
        return jsonify({"error": "Missing parameters"}), 400
    
    unique_filename = data['unique_filename']
    date_folder = data['date_folder']
    file_id = data['file_id']
    
    # Разбираем путь даты на компоненты
    year, month, day = date_folder.split('/')
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], year, month, day, unique_filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    try:
        # Запускаем транскрибацию в отдельном потоке
        import threading
        
        def transcribe_thread():
            try:
                txt_filename = transcribe_large_audio(
                    file_path, 
                    file_id, 
                    model_size=WHISPER_MODEL or "medium"
                )
                
                # Сохраняем имя файла в прогресс
                progress = TranscriptionProgress.get_instance(file_id)
                progress.txt_filename = txt_filename
                
            except Exception as e:
                logging.error(f"Transcription thread error: {str(e)}")
        
        # Запускаем транскрибацию в фоновом потоке
        thread = threading.Thread(target=transcribe_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "message": "Transcription started",
            "file_id": file_id,
            "status_url": f"/api/progress/{file_id}"
        }), 202  # 202 Accepted - запрос принят на обработку
    
    except Exception as e:
        logging.error(f"Transcription error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/progress/<path:file_id>', methods=['GET'])
def get_transcription_progress(file_id):
    """Получить прогресс транскрибации"""
    try:
        progress = TranscriptionProgress.get_instance(file_id)
        
        response = {
            "file_id": file_id,
            "status": progress.status,
            "progress": progress.progress,
            "total_segments": progress.total_segments,
            "processed_segments": progress.processed_segments,
            "lines": progress.lines[-50:],  # Последние 50 строк
            "total_lines": len(progress.lines),
            "error_message": progress.error_message
        }
        
        # Если транскрибация завершена, добавляем информацию о файле
        if progress.status == "completed" and hasattr(progress, 'txt_filename'):
            # Получаем дату из file_id
            parts = file_id.split('/')
            if len(parts) >= 4:  # year/month/day/filename
                year, month, day = parts[0], parts[1], parts[2]
                response["txt_filename"] = progress.txt_filename
                response["download_url"] = f"/api/download/{year}/{month}/{day}/{progress.txt_filename}"
        
        return jsonify(response)
    
    except KeyError:
        return jsonify({
            "error": "Transcription not found",
            "file_id": file_id,
            "status": "not_found"
        }), 404


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


@app.route('/api/cleanup', methods=['POST'])
def cleanup_progress():
    """Очистка старых записей прогресса"""
    try:
        data = request.json
        file_id = data.get('file_id')
        
        if file_id:
            TranscriptionProgress.remove_instance(file_id)
            return jsonify({"message": f"Progress for {file_id} cleaned up"})
        else:
            return jsonify({"error": "file_id required"}), 400
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Создаем папку для шаблонов, если ее нет
    os.makedirs('templates', exist_ok=True)
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True
    )