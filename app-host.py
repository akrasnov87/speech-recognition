import os
import uuid
import logging
import json
from threading import Lock
import pickle
from pathlib import Path

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
    _state_file = 'transcription_state.pkl'
    
    @classmethod
    def get_instance(cls, file_id):
        with cls._lock:
            if file_id not in cls._instances:
                cls._instances[file_id] = cls(file_id)
                cls._load_state()
            return cls._instances[file_id]
    
    @classmethod
    def remove_instance(cls, file_id):
        with cls._lock:
            if file_id in cls._instances:
                del cls._instances[file_id]
                cls._save_state()
    
    @classmethod
    def _save_state(cls):
        """Сохранить состояние в файл"""
        try:
            with open(cls._state_file, 'wb') as f:
                # Сохраняем только необходимые данные
                state_data = {}
                for file_id, progress in cls._instances.items():
                    state_data[file_id] = {
                        'lines': progress.lines,
                        'status': progress.status,
                        'progress': progress.progress,
                        'total_segments': progress.total_segments,
                        'estimated_total': progress.estimated_total,
                        'processed_segments': progress.processed_segments,
                        'error_message': progress.error_message,
                        'start_time': progress.start_time,
                        'end_time': progress.end_time,
                        'last_update_time': progress.last_update_time,
                        'txt_filename': getattr(progress, 'txt_filename', None),
                        'original_filename': getattr(progress, 'original_filename', None),
                        'file_size_mb': getattr(progress, 'file_size_mb', None),
                        'upload_time': getattr(progress, 'upload_time', None)
                    }
                pickle.dump(state_data, f)
        except Exception as e:
            logging.error(f"Error saving state: {str(e)}")
    
    @classmethod
    def _load_state(cls):
        """Загрузить состояние из файла"""
        try:
            if os.path.exists(cls._state_file):
                with open(cls._state_file, 'rb') as f:
                    state_data = pickle.load(f)
                    for file_id, data in state_data.items():
                        if file_id not in cls._instances:
                            progress = cls(file_id)
                            progress.lines = data.get('lines', [])
                            progress.status = data.get('status', 'pending')
                            progress.progress = data.get('progress', 0)
                            progress.total_segments = data.get('total_segments', 0)
                            progress.estimated_total = data.get('estimated_total', False)
                            progress.processed_segments = data.get('processed_segments', 0)
                            progress.error_message = data.get('error_message', None)
                            progress.start_time = data.get('start_time', None)
                            progress.end_time = data.get('end_time', None)
                            progress.last_update_time = data.get('last_update_time', None)
                            
                            if data.get('txt_filename'):
                                progress.txt_filename = data['txt_filename']
                            if data.get('original_filename'):
                                progress.original_filename = data['original_filename']
                            if data.get('file_size_mb'):
                                progress.file_size_mb = data['file_size_mb']
                            if data.get('upload_time'):
                                progress.upload_time = data['upload_time']
                            
                            cls._instances[file_id] = progress
        except Exception as e:
            logging.error(f"Error loading state: {str(e)}")
    
    def __init__(self, file_id):
        self.file_id = file_id
        self.lines = []
        self.status = "pending"  # pending, processing, completed, error
        self.progress = 0
        self.total_segments = 0  # 0 означает "неизвестно"
        self.processed_segments = 0
        self.estimated_total = False  # Флаг, что total_segments - оценка
        self.error_message = None
        self.start_time = None
        self.end_time = None
        self.last_update_time = None
        self.txt_filename = None
        self.original_filename = None
        self.file_size_mb = None
        self.upload_time = None
    
    def add_line(self, line):
        self.lines.append(line)
        self.processed_segments += 1
        self.last_update_time = datetime.now()
        
        # Обновляем прогресс только если знаем общее количество
        if self.total_segments > 0:
            self.progress = int((self.processed_segments / self.total_segments) * 100)
        else:
            # Если общее количество неизвестно, прогресс можно оценить по времени
            # или просто показывать количество обработанных
            self.progress = 0
        
        # Сохраняем состояние после каждого обновления
        self._save_state_async()
    
    def update_total_segments(self, total):
        """Обновить общее количество сегментов (может быть оценкой)"""
        self.total_segments = total
        if self.processed_segments > 0 and total > 0:
            self.progress = int((self.processed_segments / total) * 100)
        self._save_state_async()
    
    def set_upload_info(self, original_filename, file_size_mb):
        """Сохранить информацию о загруженном файле"""
        self.original_filename = original_filename
        self.file_size_mb = file_size_mb
        self.upload_time = datetime.now()
        self._save_state_async()
    
    def _save_state_async(self):
        """Асинхронно сохранить состояние"""
        def save():
            try:
                self.__class__._save_state()
            except Exception as e:
                logging.error(f"Async save error: {str(e)}")
        
        import threading
        thread = threading.Thread(target=save)
        thread.daemon = True
        thread.start()
    
    def to_dict(self):
        result = {
            "file_id": self.file_id,
            "status": self.status,
            "progress": self.progress,
            "total_segments": self.total_segments,
            "estimated_total": self.estimated_total,
            "processed_segments": self.processed_segments,
            "lines": self.lines[-50:],  # Последние 50 строк
            "total_lines": len(self.lines),
            "error_message": self.error_message,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "last_update_time": self.last_update_time.isoformat() if self.last_update_time else None,
            "duration": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None
        }
        
        # Добавляем информацию о файле, если она есть
        if self.original_filename:
            result["original_filename"] = self.original_filename
            result["file_size_mb"] = self.file_size_mb
            result["upload_time"] = self.upload_time.isoformat() if self.upload_time else None
        
        # Если транскрибация завершена, добавляем информацию о файле
        if self.status == "completed" and self.txt_filename:
            result["txt_filename"] = self.txt_filename
            # Получаем дату из file_id
            parts = self.file_id.split('/')
            if len(parts) >= 4:  # year/month/day/filename
                year, month, day = parts[0], parts[1], parts[2]
                result["download_url"] = f"/api/download/{year}/{month}/{day}/{self.txt_filename}"
        
        return result


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
            progress.txt_filename = f'{FILE_NAME}.txt'
            progress.end_time = datetime.now()
            
            # Сохраняем финальное состояние
            progress._save_state_async()
            
        return f'{FILE_NAME}.txt'
    
    except Exception as e:
        progress.status = "error"
        progress.error_message = str(e)
        progress.end_time = datetime.now()
        progress._save_state_async()
        logging.error(f"Transcription error for {file_id}: {str(e)}")
        raise


app = Flask(__name__)

# Конфигурация для загрузки файлов
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB
app.config['ALLOWED_EXTENSIONS'] = {'webm', 'mkv', 'mp4', 'jpg', 'mp3', 'wav', 'm4a'}

# Создаем папку для загрузок, если ее нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Загружаем сохраненное состояние при запуске
TranscriptionProgress._load_state()


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


@app.route('/api/session/restore', methods=['GET'])
def restore_session():
    """Восстановить сессию пользователя"""
    try:
        # Получаем все активные транскрибации
        active_transcriptions = []
        
        for file_id, progress in TranscriptionProgress._instances.items():
            if progress.status in ['processing', 'completed', 'pending']:
                # Проверяем, существует ли еще файл
                parts = file_id.split('/')
                if len(parts) >= 4:
                    year, month, day, filename = parts[0], parts[1], parts[2], parts[3]
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], year, month, day, filename)
                    
                    if os.path.exists(file_path):
                        active_transcriptions.append(progress.to_dict())
                    else:
                        # Файл был удален, очищаем прогресс
                        TranscriptionProgress.remove_instance(file_id)
        
        return jsonify({
            "active_transcriptions": active_transcriptions,
            "total": len(active_transcriptions)
        })
    except Exception as e:
        logging.error(f"Session restore error: {str(e)}")
        return jsonify({"error": str(e)}), 500


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
        
        # Сохраняем информацию о загруженном файле в прогресс
        progress = TranscriptionProgress.get_instance(file_id)
        progress.set_upload_info(filename, round(file_size / (1024 * 1024), 2))
        progress.status = "pending"
        
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
        # Обновляем статус прогресса
        progress = TranscriptionProgress.get_instance(file_id)
        progress.status = "processing"
        progress.start_time = datetime.now()
        
        # Запускаем транскрибацию в отдельном потоке
        import threading
        
        def transcribe_thread():
            try:
                txt_filename = transcribe_large_audio(
                    file_path, 
                    file_id, 
                    model_size=WHISPER_MODEL or "medium"
                )
                
                logging.info(f"Transcription completed for {file_id}: {txt_filename}")
                
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


# В начале каждой функции обработки запроса добавляем лог
@app.route('/api/progress/<path:file_id>', methods=['GET'])
def get_transcription_progress(file_id):
    """Получить прогресс транскрибации"""
    logging.debug(f"Progress request for: {file_id}")
    
    try:
        progress = TranscriptionProgress.get_instance(file_id)
        
        # Логируем состояние
        logging.debug(f"Progress state for {file_id}: status={progress.status}, "
                     f"processed={progress.processed_segments}, total={progress.total_segments}")
        
        response = progress.to_dict()
        
        # Добавляем timestamp для клиента
        response["server_time"] = datetime.now().isoformat()
        
        return jsonify(response)
    
    except KeyError:
        logging.warning(f"Progress not found for: {file_id}")
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


# Обновляем функцию cleanup_progress
@app.route('/api/cleanup', methods=['POST'])
def cleanup_progress():
    """Очистка старых записей прогресса"""
    try:
        data = request.json
        file_id = data.get('file_id')
        force = data.get('force', False)  # Новый параметр для принудительной очистки
        
        if file_id:
            # Если запрашивается принудительная очистка, удаляем сразу
            if force:
                TranscriptionProgress.remove_instance(file_id)
                return jsonify({
                    "message": f"Progress for {file_id} force cleaned up",
                    "force": True
                })
            
            # Стандартная логика: удаляем только если транскрибация завершена
            progress = TranscriptionProgress.get_instance(file_id)
            if progress.status in ['completed', 'error']:
                TranscriptionProgress.remove_instance(file_id)
                return jsonify({
                    "message": f"Progress for {file_id} cleaned up",
                    "status": progress.status
                })
            else:
                return jsonify({
                    "message": f"Progress for {file_id} not cleaned up - status is {progress.status}",
                    "status": progress.status,
                    "skip": True
                })
        else:
            # Очистка старых записей (старше 24 часов)
            import time
            current_time = datetime.now()
            
            to_remove = []
            for fid, progress in TranscriptionProgress._instances.items():
                if progress.last_update_time:
                    time_diff = (current_time - progress.last_update_time).total_seconds()
                    if time_diff > 24 * 3600:  # 24 часа
                        to_remove.append(fid)
                elif progress.status in ['completed', 'error']:
                    # Удаляем завершенные без времени обновления
                    to_remove.append(fid)
            
            for fid in to_remove:
                TranscriptionProgress.remove_instance(fid)
            
            return jsonify({
                "message": f"Cleaned up {len(to_remove)} old entries",
                "cleaned": to_remove
            })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Добавляем новый эндпоинт для полного сброса
@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Полный сброс сессии для текущего пользователя"""
    try:
        data = request.json
        file_id = data.get('file_id')
        
        if file_id:
            # Удаляем конкретный файл
            TranscriptionProgress.remove_instance(file_id)
            
            # Также удаляем файл с диска
            parts = file_id.split('/')
            if len(parts) >= 4:
                year, month, day, filename = parts[0], parts[1], parts[2], parts[3]
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], year, month, day, filename)
                txt_path = file_path.replace(os.path.splitext(file_path)[1], '.txt')
                
                # Удаляем файлы если они существуют
                files_removed = []
                if os.path.exists(file_path):
                    os.remove(file_path)
                    files_removed.append(file_path)
                
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                    files_removed.append(txt_path)
                
                return jsonify({
                    "message": f"Session for {file_id} fully reset",
                    "files_removed": files_removed
                })
        
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