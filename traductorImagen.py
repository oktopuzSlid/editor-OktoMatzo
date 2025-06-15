import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Cargar modelo de detección (YOLOv8)
detector = YOLO("yolov8n.pt")  # Puedes cambiar a yolov8s.pt si tienes mejor GPU

# Cargar modelo de captioning (BLIP)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
captioner = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Activar cámara
cap = cv2.VideoCapture(0)

print("Presiona 'd' para describir lo que se ve.")
print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Cámara en tiempo real", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("d"):
        # Ejecutar YOLO sobre el frame
        results = detector(frame)[0]

        descriptions = []

        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                class_name = detector.names[cls_id]

                # Coordenadas del objeto
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Recortar el objeto
                obj_crop = frame[y1:y2, x1:x2]
                if obj_crop.size == 0:
                    continue

                # Convertir a PIL
                obj_image = Image.fromarray(cv2.cvtColor(obj_crop, cv2.COLOR_BGR2RGB))

                # Describir con BLIP
                inputs = processor(obj_image, return_tensors="pt")
                out = captioner.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)

                descriptions.append(f"{class_name}: {caption}")

        # Mostrar todas las descripciones
        print("\n🔍 Descripción detallada de la escena:")
        for desc in descriptions:
            print("•", desc)
        print("")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()