import cv2
import numpy as np
from ultralytics import YOLO

# Cargar modelo YOLOv8 (puedes cambiar a yolov8s.pt, yolov8m.pt, etc. según tu hardware)
model = YOLO("yolov8n.pt")  # nano = rápido

# Iniciar cámara
cap = cv2.VideoCapture(0)

# Modo inicial
mode = "normal"

print("Presiona 1 para modo normal (detección).")
print("Presiona 2 para aplicar máscara verde sobre personas.")
print("Presiona 3 para mostrar solo personas (fondo oculto).")
print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detección con mínimo nivel de confianza
    results = model(frame, conf=0.5)
    boxes = results[0].boxes

    # Teclas para cambiar de modo
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("1"):
        mode = "normal"
    elif key == ord("2"):
        mode = "mask"
    elif key == ord("3"):
        mode = "person_only"

    if mode == "normal":
        annotated = results[0].plot()
        cv2.imshow("YOLOv8 - Detección normal", annotated)

    elif mode == "mask":
        overlay = frame.copy()
        alpha = 0.5  # Transparencia

        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Persona
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)  # Verde

        # Combinar frame y máscara
        frame_masked = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.imshow("YOLOv8 - Máscara sobre personas", frame_masked)

    elif mode == "person_only":
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Persona
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Extraer solo personas
        people = cv2.bitwise_and(frame, frame, mask=mask)

        # Fondo artificial (gris oscuro)
        bg = np.full_like(frame, (50, 50, 50))
        inv_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(bg, bg, mask=inv_mask)

        # Unir personas + fondo
        final = cv2.add(people, background)
        cv2.imshow("YOLOv8 - Solo personas visibles", final)

cap.release()
cv2.destroyAllWindows()
