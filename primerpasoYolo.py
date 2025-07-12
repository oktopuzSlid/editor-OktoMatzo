import cv2
import numpy as np
import time
import multiprocessing as mp
import threading
from MODELS.loadModels import get_yolo_model
from ANALYZE.analyzeObjects import analyze_objects

model = get_yolo_model()

def apply_effects(frame, object_effects):
    processed_frame = frame.copy()
    results = model(frame, verbose=False)[0]
    boxes = results.boxes
    masks = results.masks.data.cpu().numpy() if results.masks is not None else []

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        class_name = model.model.names[cls_id]

        if i < len(masks):
            mask_raw = masks[i]
            mask_resized = cv2.resize(mask_raw, (frame.shape[1], frame.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
        else:
            mask_binary = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            mask_binary[y1:y2, x1:x2] = 1

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
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(processed_frame, contours, -1, (0, 255, 255), 2)

    return processed_frame

def webcam_stream_reader(camera_index, queue):
    cap = cv2.VideoCapture(camera_index)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reconectando webcam...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(camera_index)
            continue
        if not queue.full():
            queue.put(frame)

def prompt_effect_for_class(class_name, caption):
    print(f"\n🧠 Objeto detectado: {class_name} - {caption}")
    print("¿Qué acción quieres aplicar a esta clase?")
    print("  [1] Eliminar (filtro negro)")
    print("  [2] Resaltar contorno")
    print("  [3] Pintar color personalizado")
    print("  [Enter] Ignorar\n")
    choice = input(f"Opción para '{class_name}': ").strip()

    if choice == "1":
        return {"action": "eliminar"}
    elif choice == "2":
        return {"action": "resaltar"}
    elif choice == "3":
        r = int(input("  🔴 R: "))
        g = int(input("  🟢 G: "))
        b = int(input("  🔵 B: "))
        return {"action": "color", "color": [b, g, r]}
    else:
        return None

def main():
    mp.freeze_support()
    frame_queue = mp.Queue(maxsize=1)
    result_queue = mp.Queue()
    analysis_process = None
    object_effects = {}

    CAMERA_INDEX = 0
    webcam_thread = threading.Thread(
        target=webcam_stream_reader,
        args=(CAMERA_INDEX, frame_queue),
        daemon=True
    )
    webcam_thread.start()

    print("Presiona 'd' para analizar, 'q' para salir.")

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get_nowait()
            processed_frame = apply_effects(frame, object_effects)

            status = "Analizando..." if analysis_process and analysis_process.is_alive() else "Listo"
            cv2.putText(processed_frame, f"Estado: {status}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Webcam - Detección YOLO", processed_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("d") and (analysis_process is None or not analysis_process.is_alive()):
                print("\n🔍 Iniciando análisis...")
                analysis_process = mp.Process(
                    target=analyze_objects,
                    args=(frame.copy(), result_queue)
                )
                analysis_process.start()

            elif key == ord("q"):
                break

            if not result_queue.empty():
                objects_info = result_queue.get()
                for obj in objects_info:
                    class_name = obj['class_name']
                    if class_name not in object_effects:
                        effect = prompt_effect_for_class(class_name, obj['caption'])
                        if effect:
                            object_effects[class_name] = effect

    if analysis_process and analysis_process.is_alive():
        analysis_process.terminate()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
