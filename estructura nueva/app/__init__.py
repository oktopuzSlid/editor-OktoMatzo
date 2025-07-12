"""
Paquete principal de la aplicación - Contiene la lógica central del editor de video
"""

from .video_editor import VideoEditor
from .video_source_manager import VideoSourceManager, VideoSourceType

# Exporta los símbolos principales
__all__ = [
    'VideoEditor',
    'VideoSourceManager',
    'VideoSourceType'
]

# Inicialización del paquete
print(f"Paquete app cargado correctamente ({__name__})")