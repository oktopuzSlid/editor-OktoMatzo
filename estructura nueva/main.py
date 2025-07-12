"""
Módulo principal del editor de video con YOLO.
Coordina todos los componentes del sistema.
"""

import multiprocessing as mp
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer

from config.settings import VIDEO_SOURCES

from app.video_editor import VideoEditor
from web.server import start_web_server

def main():
    # Configuración para Windows
    mp.freeze_support()
    
    # Iniciar servidor Flask en hilo separado
    flask_thread = threading.Thread(
        target=start_web_server,
        daemon=True
    )
    
    flask_thread.start()
    
    # Crear editor de video
    editor = VideoEditor()
    
    try:
        # Configurar fuente predeterminada ANTES de ejecutar
        success, message = editor.set_video_source(
            source_type='webcam',
            device_index=VIDEO_SOURCES['webcam']['device_index']
        )
        
        if not success:
            raise RuntimeError(f"No se pudo inicializar la fuente de video: {message}")
        
        # Ahora sí ejecutar
        editor.run()
        
    except KeyboardInterrupt:
        print("\nAplicación terminada por el usuario")
    except Exception as e:
        print(f"\nError crítico: {str(e)}")
    finally:
        editor.cleanup()
        

if __name__ == "__main__":
    main()