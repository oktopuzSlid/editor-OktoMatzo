import cv2
from PIL import Image
from models.yolo.loader import get_yolo_model, get_blip_processor, get_blip_captioner

def analyze_objects(frame, result_queue):
    """Función para análisis en proceso separado"""
    try:
        # Cargar modelos dentro del proceso hijo
        model = get_yolo_model()
        processor = get_blip_processor()
        captioner = get_blip_captioner()
        
        # Convertir a RGB (YOLO espera RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar objetos
        results = model(frame_rgb)[0]
        boxes = results.boxes
        
        objects_info = []
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            # Recorte del objeto
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_crop = frame[y1:y2, x1:x2]
            if obj_crop.size == 0:
                continue

            # Descripción con BLIP
            obj_pil = Image.fromarray(cv2.cvtColor(obj_crop, cv2.COLOR_BGR2RGB))
            inputs = processor(obj_pil, return_tensors="pt")
            out = captioner.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            objects_info.append({
                'class_name': class_name,
                'caption': caption
            })
            
        result_queue.put(objects_info)
        
    except Exception as e:
        print(f"Error en análisis: {e}")
        result_queue.put([])
