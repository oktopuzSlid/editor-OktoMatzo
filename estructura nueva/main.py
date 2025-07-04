"""
Módulo principal del editor de video con YOLO.
Coordina todos los componentes del sistema.
"""

import multiprocessing as mp
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer

from config.settings import RTSP_URL
from app.camera_manager import RTSPStreamReader
from app.video_editor import VideoEditor
from web.server import start_web_server

def main():
    """Función principal que inicia todos los componentes del sistema."""
    # Iniciar servidor web en hilo separado
    web_thread = threading.Thread(
        target=start_web_server,
        daemon=True
    )
    web_thread.start()
    
    # Configuración de multiprocesamiento para Windows
    mp.freeze_support()
    
    # Iniciar el editor de video
    editor = VideoEditor(RTSP_URL)
    editor.run()

if __name__ == "__main__":
    main()