from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

# Variables globales para almacenar los modelos
_yolo_model = None
_blip_processor = None
_blip_captioner = None

def get_yolo_model(model_path="yolov8n-seg.pt"):
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(model_path)
    return _yolo_model

def get_blip_processor():
    global _blip_processor
    if _blip_processor is None:
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    return _blip_processor

def get_blip_captioner():
    global _blip_captioner
    if _blip_captioner is None:
        _blip_captioner = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return _blip_captioner