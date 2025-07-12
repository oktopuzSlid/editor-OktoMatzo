import cv2
import time
import threading
from pathlib import Path
from enum import Enum, auto

class VideoSourceType(Enum):
    IP_CAMERA = auto()
    WEBCAM = auto()
    FILE = auto()

class VideoSourceManager:
    def __init__(self, source_type, **kwargs):
        self.source_type = source_type
        self.cap = None
        self.current_source = ""
        self.lock = threading.Lock()
        
        if source_type == VideoSourceType.IP_CAMERA:
            self._init_ip_camera(kwargs.get('rtsp_url'))
        elif source_type == VideoSourceType.WEBCAM:
            self._init_webcam(kwargs.get('device_index', 0))
        elif source_type == VideoSourceType.FILE:
            self._init_file_source(kwargs.get('file_path'))
    
    def _init_ip_camera(self, rtsp_url):
        """Configura fuente RTSP con parámetros optimizados"""
        self.current_source = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        # Parámetros críticos para RTSP
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            raise ConnectionError(f"No se pudo conectar a {rtsp_url}")
    
    def _init_webcam(self, device_index):
        """Configura cámara web local"""
        self.current_source = f"Webcam {device_index}"
        self.cap = cv2.VideoCapture(device_index)
        
        # Configurar resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            raise ConnectionError(f"No se pudo abrir webcam {device_index}")
    
    def _init_file_source(self, file_path):
        """Configura archivo de video como fuente"""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        self.current_source = file_path
        self.cap = cv2.VideoCapture(file_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"No se pudo abrir el archivo: {file_path}")
    
    def get_frame(self):
        """Obtiene el frame actual con manejo de errores"""
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return None, "Fuente no disponible"
            
            ret, frame = self.cap.read()
            
            # Para archivos de video, reiniciar al final
            if not ret and self.source_type == VideoSourceType.FILE:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            
            return frame if ret else None, ""
    
    def release(self):
        """Libera los recursos"""
        with self.lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.cap = None
    
    def set_video_source(self, source_type, **kwargs):
        """Configura la fuente de video de manera segura"""
        try:
            if source_type not in ['ip_camera', 'webcam', 'file']:
                return False, "Tipo de fuente no válido"
            
            # Liberar fuente existente
            if hasattr(self, 'video_source') and self.video_source is not None:
                self.video_source.release()
            
            # Crear nueva fuente
            self.video_source = VideoSourceManager()
            
            if source_type == 'ip_camera':
                rtsp_url = kwargs.get('rtsp_url', VIDEO_SOURCES['ip_camera']['url'])
                self.video_source.configure(VideoSourceType.IP_CAMERA, rtsp_url=rtsp_url)
            elif source_type == 'webcam':
                device_index = kwargs.get('device_index', 0)
                self.video_source.configure(VideoSourceType.WEBCAM, device_index=device_index)
            elif source_type == 'file':
                file_path = kwargs.get('file_path')
                if not file_path:
                    return False, "Ruta de archivo no proporcionada"
                self.video_source.configure(VideoSourceType.FILE, file_path=file_path)
                
            return True, f"Fuente {source_type} configurada correctamente"
            
        except Exception as e:
            return False, f"Error al configurar fuente: {str(e)}"
    
    def get_source_info(self):
        """Devuelve información sobre la fuente actual"""
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 0
        return {
            'type': self.source_type.name,
            'source': self.current_source,
            'fps': fps,
            'is_active': self.cap is not None and self.cap.isOpened()
        }