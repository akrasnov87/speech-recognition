import os
import uuid
import logging

from flask import Flask, send_file, jsonify, request, abort
from werkzeug.utils import secure_filename
from datetime import datetime
from faster_whisper import WhisperModel
from dotenv import load_dotenv

ENV_PATH=os.environ.get('ENV_PATH') if os.environ.get('ENV_PATH') != None else '.env'

load_dotenv(ENV_PATH)

logging.basicConfig(level=logging.DEBUG if os.getenv('LOG_LEVEL') == 'DEBUG' else logging.INFO)

logging.info(f'ENV_PATH={ENV_PATH}')

WHISPER_MODEL='medium' if os.environ.get('WHISPER_MODEL') != None else os.environ.get('WHISPER_MODEL')

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
        
    DIR=os.path.dirname(input_path)
    FILE_NAME=os.path.splitext(os.path.basename(input_path))[0]
    TXT_FILE_PATH=f"{os.path.join(DIR, f'{FILE_NAME}.txt')}"

    with open(TXT_FILE_PATH, "w", encoding="utf-8") as f:
        for seg in segments:
            line = f"[{seconds_to_hms(seg.start)} → {seconds_to_hms(seg.end)}]: {seg.text}\n"
            f.write(line)
            logging.info(line.strip())
    
    return f'{FILE_NAME}.txt'

app = Flask(__name__)

# Конфигурация для загрузки файлов
app.config['UPLOAD_FOLDER'] = 'uploads'  # Папка для сохранения файлов
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # Максимальный размер файла 1 Gb
app.config['ALLOWED_EXTENSIONS'] = {'webm', 'mkv', 'mp4', 'jpg'}

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
    # Формат: ГГГГ/ММ/ДД
    now = datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)  # Добавляем ведущий ноль
    day = str(now.day).zfill(2)  # Добавляем ведущий ноль
    
    # Создаем путь папки
    date_folder = os.path.join(year, month, day)
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], date_folder)
    
    # Создаем папки, если их нет
    os.makedirs(full_path, exist_ok=True)
    
    return full_path, date_folder

def allowed_file(filename):
    """Проверка разрешенных расширений файлов"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/health_check', methods=['GET'])
def health_check():
    return jsonify({"status": "success"})

@app.route('/upload', methods=['POST'])
def upload_file():
    """Эндпоинт для загрузки файла"""
    # Проверяем, есть ли файл в запросе
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # Если пользователь не выбрал файл
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Безопасное имя файла
        filename = secure_filename(file.filename)
        upload_folder, _ = get_date_based_folder()
        # Сохраняем файл
        uniq_name = get_unique_filename(filename)
        file_path = os.path.join(upload_folder, uniq_name)
        file.save(file_path)
        
        return jsonify({
            "message": "File uploaded successfully",
            "filename": filename,
            "uniq_name": uniq_name,
            "file_path": file_path
        }), 200
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/files/<year>/<month>/<day>', methods=['GET'])
def list_files(year, month, day):
    """Получение списка файлов для конкретной даты"""
    # Формируем путь к папке
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], year, month, day)
    # Проверяем, существует ли папка
    if not os.path.exists(folder_path):
        return jsonify({
            "error": "Folder not found",
            "path": folder_path
        }), 404
    
    try:
        # Получаем список файлов
        files = os.listdir(folder_path)
        
        # Дополнительная информация о файлах
        file_info = []
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):  # Проверяем, что это файл, а не папка
                file_size = os.path.getsize(file_path)
                file_info.append({
                    "name": filename,
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "path": os.path.join(year, month, day, filename)
                })
        
        return jsonify({
            "year": year,
            "month": month,
            "day": day,
            "folder_path": folder_path,
            "files_count": len(file_info),
            "files": file_info
        })
    
    except PermissionError:
        return jsonify({"error": "Permission denied"}), 403
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run/<year>/<month>/<day>/<name>', methods=['GET'])
def run(year, month, day, name):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], year, month, day, name)
    txt_path = transcribe_large_audio(file_path, model_size=WHISPER_MODEL)
    return jsonify({
            "path": txt_path
    }), 200

@app.route('/download/<year>/<month>/<day>/<name>', methods=['GET'])
def download_file(year, month, day, name):
    """Скачать файл по имени"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], year, month, day, name)
        
        # Проверяем существование файла
        if not os.path.exists(file_path):
            abort(404, description="File not found")
        
        # Проверяем, что это файл, а не папка
        if not os.path.isfile(file_path):
            abort(400, description="Not a file")
        
        # Отправляем файл с указанием имени для скачивания
        return send_file(
            file_path,
            as_attachment=True,  # Принудительное скачивание
            download_name=name  # Имя файла для браузера (Flask 2.0+)
            # Для Flask < 2.0 используйте attachment_filename
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0', 
        port=5001
    )