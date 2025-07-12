# Configuraciones de video
VIDEO_SOURCES = {
    'ip_camera': {
        'url': 'rtsp://192.168.195.112:8554/camara1',
        'timeout': 30  # segundos
    },
    'webcam': {
        'device_index': 0,  # Normalmente 0 para la cámara predeterminada
        'resolution': (1280, 720)
    },
    'file': {
        'default_path': './uploads/videos',
        'allowed_extensions': ['.mp4', '.avi', '.mov']
    }
}