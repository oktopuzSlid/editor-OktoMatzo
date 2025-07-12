"""
Paquete web - Contiene el servidor Flask y componentes de la interfaz
"""

from .server import app, start_flask_server

__all__ = [
    'app',
    'start_flask_server'
]

# Configuración básica de Flask
import os
os.environ['FLASK_ENV'] = 'development'