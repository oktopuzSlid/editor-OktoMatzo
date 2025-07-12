import cv2
import numpy as np
import time
import threading
from models.yolo.loader import get_yolo_model

from app.video_source_manager import VideoSourceManager, VideoSourceType

class VideoEditor:
    def __init__(self):
        self.model = get_yolo_model()
        self.object_effects = {}
        self.video_source = None  # Se configurará después
        self.current_frame = None
        self.frame_lock = threading.Lock()
    
    def set_video_source(self, source_type, **kwargs):
        """Configura la fuente de video"""
        if self.video_source:
            self.video_source.release()
        
        try:
            self.video_source = VideoSourceManager(source_type, **kwargs)
            return True, "Fuente configurada correctamente"
        except Exception as e:
            return False, str(e)
    
    def process_frame(self, frame):
        """Procesa un frame aplicando efectos YOLO"""
        if frame is None:
            return None
        
        processed_frame = frame.copy()
        results = self.model(frame, verbose=False)[0]
        
        # ... (resto del procesamiento YOLO igual que antes)
        
        return processed_frame
    
    def run(self):
        """Bucle principal de procesamiento"""
        if not hasattr(self, 'video_source') or self.video_source is None:
            raise RuntimeError("Video source no ha sido configurado. Llama a set_video_source() primero")
        
        self.running = True
        
        try:
            while self.running:
                frame, error = self.video_source.get_frame()
                
                if frame is None:
                    if error:
                        print(f"Error: {error}")
                    time.sleep(0.1)
                    continue
                
                processed = self.process_frame(frame)
                
                with self.frame_lock:
                    _, buffer = cv2.imencode('.jpg', processed)
                    self.current_frame = buffer.tobytes()
                
                time.sleep(0.033)  # ~30 FPS
        
        except Exception as e:
            print(f"Error en el bucle principal: {str(e)}")
        finally:
            self.cleanup()
    
    
    def analyze_frame(self, frame):
        """Analiza un frame y devuelve objetos detectados"""
        results = self.model(frame, verbose=False)[0]
        detected_objects = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.model.names[cls_id]
            detected_objects.append({
                "class_name": class_name,
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })
            
        return detected_objects
    
    def update_effect(self, class_name, effect_type, color=None):
        """Actualiza los efectos para una clase de objeto"""
        effect = {"action": effect_type}
        if effect_type == "color" and color:
            effect["color"] = (color[2], color[1], color[0])  # RGB to BGR
        
        self.object_effects[class_name] = effect
    
    def _get_mask(self, idx, masks, box, frame_shape):
        """Genera máscara binaria para un objeto"""
        if masks and idx < len(masks.data):
            mask_raw = masks.data[idx].cpu().numpy()
            mask_resized = cv2.resize(mask_raw, (frame_shape[1], frame_shape[0]))
            return (mask_resized > 0.5).astype(np.uint8)
        else:
            mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            mask[y1:y2, x1:x2] = 1
            return mask
    
    def _apply_effect(self, frame, mask, effect):
        """Aplica un efecto específico a una región del frame"""
        if effect['action'] == 'remove':
            frame[mask == 1] = 0
        elif effect['action'] == 'color':
            color = np.array(effect['color'], dtype=np.uint8)
            color_layer = np.zeros_like(frame, dtype=np.uint8)
            color_layer[:] = color
            frame = np.where(mask[:, :, None] == 1, color_layer, frame)
        elif effect['action'] == 'highlight':
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)
        return frame
    
    def cleanup(self):
        """Limpia recursos de manera segura"""
        if hasattr(self, 'video_source') and self.video_source is not None:
            self.video_source.release()
        self.running = False