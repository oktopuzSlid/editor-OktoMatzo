"""
Módulo para manejo de streams de cámara RTSP/IP.
Incluye reconexión automática y manejo de errores.
"""

import cv2
import time
import queue
import threading

class RTSPStreamReader:
    """Lee un stream RTSP en un hilo separado con reconexión automática."""
    
    def __init__(self, rtsp_url, max_retries=5):
        """
        Inicializa el lector de stream.
        
        Args:
            rtsp_url (str): URL del stream RTSP
            max_retries (int): Intentos máximos de reconexión
        """
        self.rtsp_url = rtsp_url
        self.max_retries = max_retries
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None

    def start(self):
        """Inicia el hilo de captura de frames."""
        self.running = True
        self.thread = threading.Thread(
            target=self._capture_frames,
            daemon=True
        )
        self.thread.start()

    def stop(self):
        """Detiene la captura de frames."""
        self.running = False
        if self.thread:
            self.thread.join()

    def _capture_frames(self):
        """Función interna que captura frames del stream."""
        cap = None
        retry_count = 0
        
        while self.running and retry_count < self.max_retries:
            try:
                # Configuración óptima para streams RTSP
                cap = cv2.VideoCapture(self.rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Actualizar el frame más reciente
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)
                
            except Exception as e:
                print(f"Error en captura de frames: {str(e)}")
            
            finally:
                if cap:
                    cap.release()
                
                if self.running:
                    retry_count += 1
                    print(f"Reconectando... Intento {retry_count}/{self.max_retries}")
                    time.sleep(2)

    def get_latest_frame(self):
        """Obtiene el frame más reciente del stream."""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None