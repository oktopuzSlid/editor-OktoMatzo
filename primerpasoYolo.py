import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import multiprocessing as mp
import time

# Cargar modelos en el proceso principal
model = YOLO("yolov8n-seg.pt")  # Modelo de segmentación

def analyze_objects(frame, result_queue):
    """Función para análisis en proceso separado"""
    try:
        # Cargar modelos BLIP dentro del proceso hijo
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        captioner = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
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

def apply_effects(frame, object_effects):
    """Aplica los efectos configurados a los objetos detectados en el frame"""
    processed_frame = frame.copy()
    
    # Detectar objetos en el frame actual
    results = model(frame, verbose=False)[0]
    boxes = results.boxes
    masks = results.masks.data.cpu().numpy() if results.masks is not None else []
    
    # Aplicar efectos a cada objeto detectado
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        
        # Obtener máscara si existe
        if i < len(masks):
            mask_raw = masks[i]
            mask_resized = cv2.resize(mask_raw, (frame.shape[1], frame.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
        else:
            # Si no hay máscara, usar bbox como aproximación
            mask_binary = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            mask_binary[y1:y2, x1:x2] = 1
        
        # Aplicar efectos configurados para esta clase
        if class_name in object_effects:
            effect = object_effects[class_name]
            
            if effect['action'] == 'eliminar':
                processed_frame[mask_binary == 1] = 0
            elif effect['action'] == 'color':
                color = np.array(effect['color'], dtype=np.uint8)
                color_layer = np.zeros_like(processed_frame, dtype=np.uint8)
                color_layer[:] = color
                processed_frame = np.where(mask_binary[:, :, None] == 1, color_layer, processed_frame)
            elif effect['action'] == 'resaltar':
                contours, _ = cv2.findContours(mask_binary, 
                                              cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(processed_frame, contours, -1, (0, 255, 255), 2)
    
    return processed_frame

def main():
    # Iniciar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    # Variables para multiprocesamiento
    result_queue = mp.Queue()
    analysis_process = None
    
    # Efectos guardados por clase: {class_name: effect_dict}
    object_effects = {}
    
    print("Presiona 'd' para analizar el frame y aplicar efectos.")
    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Aplicar efectos a los objetos detectados en este frame
        processed_frame = apply_effects(frame, object_effects)
        
        # Mostrar estado de análisis
        status = "Analizando..." if analysis_process else "Listo"
        cv2.putText(processed_frame, f"Estado: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Cámara en tiempo real", processed_frame)
        key = cv2.waitKey(1) & 0xFF

        # Manejar teclas
        if key == ord("d"):
            if not analysis_process:  # Evitar múltiples análisis simultáneos
                print("\n🔍 Iniciando análisis en segundo plano...")
                analysis_process = mp.Process(
                    target=analyze_objects,
                    args=(frame.copy(), result_queue)
                )
                analysis_process.start()
                
        elif key == ord("q"):
            break

        # Comprobar si el análisis ha terminado
        if analysis_process and not result_queue.empty():
            objects_info = result_queue.get()
            analysis_process.join()
            analysis_process = None
            
            if objects_info:
                print("\n✅ Análisis completado. Por favor responde las preguntas:")
                
                for obj_info in objects_info:
                    print(f"\n📦 Objeto: {obj_info['class_name']} — {obj_info['caption']}")
                    
                    # Preguntar al usuario
                    action = input("¿Qué quieres hacer con todos los objetos de esta clase? (eliminar/color/resaltar/nada): ").strip().lower()
                    
                    if action in ["eliminar", "color", "resaltar"]:
                        effect = {'action': action}
                        
                        if action == "color":
                            color_input = input("Color (R,G,B ej: 0,255,0): ").strip()
                            try:
                                r, g, b = map(int, color_input.split(","))
                                effect['color'] = (b, g, r)  # OpenCV: BGR
                            except:
                                print("Color inválido. Usando verde.")
                                effect['color'] = (0, 255, 0)
                        
                        # Guardar por clase (sobrescribe si ya existía)
                        object_effects[obj_info['class_name']] = effect
            
            print("\n🎨 Efectos aplicados. Mostrando video...")

    # Limpieza final
    cap.release()
    cv2.destroyAllWindows()
    if analysis_process:
        analysis_process.terminate()

if __name__ == "__main__":
    # Solución para Windows (evitar errores de multiprocesamiento)
    mp.freeze_support()
    main()