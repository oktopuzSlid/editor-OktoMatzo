import cv2 

from ultralytics import YOLO 

  

# Cargar el modelo YOLOv8 preentrenado (puede ser yolov8n, yolov8s, yolov8m, yolov8l, etc.) 

model = YOLO("yolov8n.pt")  # n = nano (más rápido, menos preciso) 

  

# Abrir la cámara (0 es la webcam por defecto) 

cap = cv2.VideoCapture(0) 

  

while True: 

    ret, frame = cap.read() 

    if not ret: 

        break 

  

    # Realizar predicción 

    results = model(frame) 

  

    # Dibujar los resultados 

    annotated_frame = results[0].plot() 

  

    # Mostrar la imagen con las detecciones 

    cv2.imshow("YOLOv8 Detección", annotated_frame) 

  

    # Salir con la tecla 'q' 

    if cv2.waitKey(1) & 0xFF == ord("q"): 

        break 

  

cap.release() 

cv2.destroyAllWindows() 