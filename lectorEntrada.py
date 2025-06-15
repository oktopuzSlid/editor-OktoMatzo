
import cv2
from modelo import INPUT_TYPE, VIDEO_PATH, IMAGE_PATH

def get_video_capture():
    if INPUT_TYPE == 'webcam':
        return cv2.VideoCapture(0)
    elif INPUT_TYPE == 'video':
        return cv2.VideoCapture(VIDEO_PATH)
    elif INPUT_TYPE == 'image':
        img = cv2.imread(IMAGE_PATH)
        return None, img  # No se necesita captura, solo imagen fija
    else:
        raise ValueError("INPUT_TYPE no válido. Usa 'webcam', 'video' o 'image'.")