from flask import Flask, render_template, request, Response, jsonify
import cv2
import subprocess
import threading
import time
import logging
import numpy as np
from ultralytics import YOLO
import datetime

app = Flask(__name__)

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales
current_frame = None
camera_process = None
camera_active = False
rtsp_url = ""
model = YOLO("yolov8n.pt")
detection_history = []
frame_counter = 0
last_analysis_time = 0
ANALYSIS_INTERVAL = 5  # segundos

def rtsp_to_mjpeg(rtsp_url):
    global current_frame, camera_active
    
    # Comando FFmpeg para convertir RTSP a MJPEG
    command = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',  # Usar TCP para mejor estabilidad
        '-i', rtsp_url,
        '-q:v', '2',               # Calidad de video
        '-update', '1',
        '-f', 'mjpeg',
        'pipe:1'
    ]
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**6  # Buffer grande para evitar bloqueos
    )
    
    try:
        while camera_active:
            # Leer el tamaño del frame (prefijo de 8 bytes)
            header = process.stdout.read(8)
            if len(header) < 8:
                break
                
            # Leer el frame JPEG
            size = int.from_bytes(header, byteorder='big')
            frame_data = process.stdout.read(size)
            
            if len(frame_data) < size:
                break
                
            # Actualizar el frame actual
            current_frame = frame_data
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()

def generate_frames():
    global current_frame, camera_active
    while camera_active:
        if current_frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
        time.sleep(0.05)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test_route():
    return "¡El servidor está funcionando correctamente!", 200

@app.route('/start_stream', methods=['POST'])  # CORRECCIÓN: Cambiado a /start_stream
def start_stream():
    global camera_active, rtsp_url
    
    if camera_active:
        return jsonify({"error": "Stream ya activo"}), 400
    
    rtsp_url = request.form['url']
    
    if not rtsp_url:
        return jsonify({"error": "URL requerida"}), 400
    
    camera_active = True
    
    # Iniciar el proceso de conversión en un hilo separado
    threading.Thread(target=rtsp_to_mjpeg, args=(rtsp_url,), daemon=True).start()
    
    return jsonify({"status": "success", "message": "Stream iniciado"}), 200

@app.route('/stop_stream', methods=['POST'])  # Cambiado a POST para consistencia
def stop_stream():
    global camera_active
    camera_active = False
    return jsonify({"status": "success", "message": "Stream detenido"}), 200

@app.route('/video_feed')
def video_feed():
    if not camera_active:
        return Response(b'', status=204)
    
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    global current_frame, detection_history, frame_counter, last_analysis_time
    
    if not current_frame:
        return jsonify({"status": "error", "message": "No hay frame disponible"}), 400
    
    # Convertir a formato OpenCV
    nparr = np.frombuffer(current_frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({"status": "error", "message": "Error decodificando frame"}), 400
    
    # Verificar intervalo mínimo
    current_time = time.time()
    if current_time - last_analysis_time < ANALYSIS_INTERVAL:
        return jsonify({
            "status": "skipped",
            "message": f"Espere {ANALYSIS_INTERVAL - int(current_time - last_analysis_time)} segundos para otro análisis"
        }), 429  # Código 429: Too Many Requests
    
    last_analysis_time = current_time
    frame_counter += 1
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Realizar detección
        results = model(frame)
        
        # Procesar resultados
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": box.xyxy[0].tolist()
                })
        
        # Guardar en historial
        detection_history.append({
            "timestamp": timestamp,
            "detections": detections
        })
        
        return jsonify({
            "status": "success",
            "timestamp": timestamp,
            "detections": detections
        }), 200
    
    except Exception as e:
        logger.error(f"Error en análisis: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_detections')
def get_detections():
    return jsonify(detection_history), 200

@app.route('/clear_detections', methods=['POST'])
def clear_detections():
    global detection_history
    detection_history = []
    return jsonify({"status": "success", "message": "Historial borrado"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, ssl_context='adhoc', threaded=True)