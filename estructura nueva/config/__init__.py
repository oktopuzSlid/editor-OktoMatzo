"""
Paquete de configuración - Contiene ajustes del sistema
"""

from .settings import VIDEO_SOURCES, YOLO_CONFIG

__all__ = [
    'VIDEO_SOURCES',
    'YOLO_CONFIG'
]

# Validación básica de configuraciones
assert isinstance(VIDEO_SOURCES, dict), "VIDEO_SOURCES debe ser un diccionario"
assert isinstance(YOLO_CONFIG, dict), "YOLO_CONFIG debe ser un diccionario"