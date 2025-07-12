import cv2
import numpy as np
import time
import multiprocessing as mp
from MODELS.loadModels import get_yolo_model
from ANALYZE.analyzeObjects import analyze_objects

# Cargar modelo YOLO en el proceso principal
model = get_yolo_model()

def apply_effects(frame, object_effects):
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
        if class_name in object_effects:
            effect = object_effects[class_name]
            
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

def rtsp_stream_reader(rtsp_url, queue):
    """Hilo dedicado para lectura RTSP robusta"""
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reconectando...")  # Reconexión automática
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(rtsp_url)
            continue
        queue.put(frame)  # Comparte frames con el hilo principal

def main():
    # Configuración RTSP
    RTSP_URL = 'rtsp://192.168.1.72:8080/h264_ulaw.sdp'
    
    # Cola para compartir frames entre hilos
    frame_queue = mp.Queue(maxsize=1)
    
    # Iniciar hilo para lectura RTSP
    rtsp_thread = threading.Thread(
        target=rtsp_stream_reader,
        args=(RTSP_URL, frame_queue),
        daemon=True
    )
    rtsp_thread.start()
    
    # Esperar primer frame
    while frame_queue.empty():
        time.sleep(0.1)
    
    # Variables para procesamiento
    result_queue = mp.Queue()
    analysis_process = None
    object_effects = {}
    
    print("Control: 'd' para analizar, 'q' para salir")

    while True:
        # Obtener frame más reciente
        if not frame_queue.empty():
            frame = frame_queue.get_nowait()
            
            # Procesar frame
            processed_frame = apply_effects(frame, object_effects)
            
            # Mostrar estado
            status = "Analizando..." if analysis_process and analysis_process.is_alive() else "Listo"
            cv2.putText(processed_frame, f"Estado: {status}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar video
            cv2.imshow("Cámara IP - Detección YOLO", processed_frame)
            
            # Manejo de teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord("d") and (analysis_process is None or not analysis_process.is_alive()):
                print("\n🔍 Iniciando análisis...")
                analysis_process = mp.Process(
                    target=analyze_objects,
                    args=(frame.copy(), result_queue)
                )
                analysis_process.start()
            elif key == ord("q"):
                break
            
            # Manejar resultados (mantener tu lógica existente)
            if analysis_process and not result_queue.empty():cv2.destroyAllWindows()
    if analysis_process and analysis_process.is_alive():
        analysis_process.terminate()
                # ... (tu código existente para manejar resultados) ...

    # Limpieza
    

from http.server import SimpleHTTPRequestHandler, HTTPServer
import threading

def run_http_server():
    port = 8000
    handler = SimpleHTTPRequestHandler
    httpd = HTTPServer(('', port), handler)
    print(f"Servidor HTTP en http://localhost:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    # Iniciar servidor HTTP en un hilo separado
    http_thread = threading.Thread(target=run_http_server)
    http_thread.daemon = True
    http_thread.start()
    
    # Solución para Windows (evitar errores de multiprocesamiento)
    mp.freeze_support()
    main()