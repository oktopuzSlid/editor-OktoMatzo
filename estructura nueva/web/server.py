"""
Servidor web Flask para la interfaz del editor de video.
"""
from flask import Flask, render_template, Response, jsonify, request

import os
from werkzeug.utils import secure_filename

import threading
import time
import cv2

try:
    from app.video_editor import VideoEditor
    from app.video_source_manager import VideoSourceType
    from config.settings import VIDEO_SOURCES
except ImportError:
    from ..app.video_editor import VideoEditor
    from ..app.video_source_manager import VideoSourceType
    from config.settings import VIDEO_SOURCES

app = Flask(__name__)
editor = VideoEditor()

# Configuración para uploads de archivos
UPLOAD_FOLDER = VIDEO_SOURCES['file']['default_path']
ALLOWED_EXTENSIONS = VIDEO_SOURCES['file']['allowed_extensions']
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/switch_source', methods=['POST'])
def switch_source():
    source_type = request.form.get('source_type')
    
    if source_type == 'ip_camera':
        success, message = editor.set_video_source(
            VideoSourceType.IP_CAMERA,
            rtsp_url=VIDEO_SOURCES['ip_camera']['url']
        )
    elif source_type == 'webcam':
        success, message = editor.set_video_source(
            VideoSourceType.WEBCAM,
            device_index=VIDEO_SOURCES['webcam']['device_index']
        )
    elif source_type == 'file':
        if 'file' not in request.files:
            return jsonify({"success": False, "message": "No se envió archivo"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "message": "Nombre de archivo vacío"})
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            success, message = editor.set_video_source(
                VideoSourceType.FILE,
                file_path=filepath
            )
        else:
            return jsonify({
                "success": False,
                "message": f"Extensión no permitida. Use: {', '.join(ALLOWED_EXTENSIONS)}"
            })
    else:
        return jsonify({"success": False, "message": "Tipo de fuente inválido"})
    
    return jsonify({"success": success, "message": message})

def video_processing_loop():
    """Bucle principal de procesamiento de video"""
    global current_frame
    while True:
        frame = VideoSourceType.get_frame()
        if frame is not None:
            processed_frame = video_processor.process_frame(frame)
            with frame_lock:
                _, buffer = cv2.imencode('.jpg', processed_frame)
                current_frame = buffer.tobytes()
        time.sleep(0.05)

@app.route('/')

def index():
    """Ruta principal que sirve la interfaz web"""
    return render_template('index.html')

@app.route('/video_feed')

def video_feed():
    """Stream de video en tiempo real"""
    def generate():
        global current_frame
        while True:
            with frame_lock:
                if current_frame:
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + 
                          current_frame + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze', methods=['POST'])

def analyze_frame():
    """Endpoint para análisis de frame actual"""
    frame = VideoSourceType.get_frame()
    if frame is None:
        return jsonify({"error": "No se pudo obtener el frame"}), 400
    
    results = video_processor.analyze_frame(frame)
    return jsonify(results)

@app.route('/update_effect', methods=['POST'])

def update_effect():
    """Endpoint para actualizar efectos"""
    data = request.json
    try:
        video_processor.update_effect(
            data['class_name'],
            data['effect_type'],
            data.get('color', [0, 255, 0])
        )
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def start_web_server():
    """Inicia el servidor web Flask"""
    # Iniciar el hilo de procesamiento de video
    video_thread = threading.Thread(target=video_processing_loop, daemon=True)
    video_thread.start()
    
    # Configuración del servidor Flask
    app.run(host='0.0.0.0', port=5000, threaded=True)