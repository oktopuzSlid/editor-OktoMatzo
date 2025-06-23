import cv2

# Dirección RTSP del servidor MediaMTX accesible por ZeroTier
rtsp_url = 'rtsp://192.168.193.112:8554/camara1'

# Abrir el stream RTSP
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("❌ Error: No se pudo conectar al stream.")
    exit()

print("✅ Conexión exitosa. Mostrando video...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se recibió frame. Saliendo...")
        break

    cv2.imshow('Stream desde MediaMTX', frame)

    # Salir con tecla ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
