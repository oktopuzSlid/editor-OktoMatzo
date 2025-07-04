import cv2
import numpy as np
import multiprocessing as mp
from flask import Flask, render_template, Response, request, jsonify
from MODELS.loadModels import get_yolo_model
from ANALYZE.analyzeObjects import analyze_objects
import time
import os

app = Flask(__name__)

# Cargar modelo YOLO en el proceso principal
model = get_yolo_model()

# Variables globales para compartir entre rutas
object_effects = {}
current_frame = None
processing = False
last_analysis = None

def apply_effects(frame, effects):
    """Aplica los efectos configurados a los objetos detectados en el frame"""
    processed_frame = frame.copy()
    
    # Detectar objetos en el frame actual
    results = model(frame, verbose=False)[0]
    boxes = results.boxes
    masks = results.masks.data.cpu().numpy() if results.masks is not None else []
    
    # Aplicar efectos a cada objeto detectado
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        class_name = model.model.names[cls_id]
        
        # Obtener máscara si existe
        if i < len(masks):
            mask_raw = masks[i]
            mask_resized = cv2.resize(mask_raw, (frame.shape[1], frame.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
        else:
            # Si no hay máscara, usar bbox como aproximación
            mask_binary = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            mask_binary[y1:y2, x1:x2] = 1
        
        # Aplicar efectos configurados para esta clase
        if class_name in effects:
            effect = effects[class_name]
            
            if effect['action'] == 'eliminar':
                processed_frame[mask_binary == 1] = 0
            elif effect['action'] == 'color':
                color = np.array(effect['color'], dtype=np.uint8)
                color_layer = np.zeros_like(processed_frame, dtype=np.uint8)
                color_layer[:] = color
                processed_frame = np.where(mask_binary[:, :, None] == 1, color_layer, processed_frame)
            elif effect['action'] == 'resaltar':
                contours, _ = cv2.findContours(mask_binary, 
                                              cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(processed_frame, contours, -1, (0, 255, 255), 2)
    
    return processed_frame

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            global current_frame, object_effects
            current_frame = frame.copy()
            
            # Aplicar efectos si hay alguno configurado
            if object_effects:
                frame = apply_effects(frame, object_effects)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze', methods=['POST'])
def analyze_frame():
    global processing, last_analysis, current_frame
    
    if processing:
        return jsonify({'status': 'busy'})
    
    processing = True
    
    # Procesamiento en segundo plano
    def analyze_in_background():
        global processing, last_analysis, object_effects
        
        result_queue = mp.Queue()
        analysis_process = mp.Process(
            target=analyze_objects,
            args=(current_frame.copy(), result_queue)
        )
        analysis_process.start()
        analysis_process.join()
        
        objects_info = result_queue.get() if not result_queue.empty() else None
        last_analysis = objects_info
        processing = False
    
    # Iniciar el análisis en un hilo separado
    import threading
    thread = threading.Thread(target=analyze_in_background)
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/get_analysis', methods=['GET'])
def get_analysis():
    global last_analysis
    if last_analysis:
        return jsonify({'status': 'ready', 'data': last_analysis})
    return jsonify({'status': 'pending'})

@app.route('/apply_effect', methods=['POST'])
def apply_effect():
    global object_effects
    data = request.json
    class_name = data.get('class_name')
    action = data.get('action')
    color = data.get('color', [0, 255, 0])  # Verde por defecto
    
    if action in ["eliminar", "color", "resaltar"]:
        effect = {'action': action}
        if action == "color":
            effect['color'] = (color[2], color[1], color[0])  # Convertir a BGR
        
        object_effects[class_name] = effect
    
    return jsonify({'status': 'success'})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = os.path.join('uploads', file.filename)
        file.save(filename)
        
        # Aquí puedes procesar el archivo con tu código YOLO
        # Por simplicidad, solo guardamos el archivo
        
        return jsonify({'status': 'success', 'filename': filename})

if __name__ == '__main__':
    # Crear directorio de uploads si no existe
    os.makedirs('uploads', exist_ok=True)
    
    # Solución para Windows (evitar errores de multiprocesamiento)
    mp.freeze_support()
    
    app.run(debug=True, threaded=True)