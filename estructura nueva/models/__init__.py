"""
Paquete de modelos - Contiene los modelos de ML y funciones relacionadas
"""

# Exporta el loader principal de YOLO
from .yolo.loader import get_yolo_model

__all__ = [
    'get_yolo_model'
]

# Advertencia si no hay GPU
import torch
if not torch.cuda.is_available():
    print("Advertencia: No se detectó GPU - El modelo correrá en CPU")