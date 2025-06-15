
import cv2
from detector import YoloDetector
from funciones import draw_normal, draw_mask, draw_person_only
from modelo import INPUT_TYPE
from lectorEntrada import get_video_capture

def main():
    detector = YoloDetector()
    if INPUT_TYPE == "image":
        _, frame = get_video_capture()
        result = detector.detect(frame)
        boxes = result.boxes

        print("Presiona 1: normal | 2: máscara verde | 3: solo personas | q: salir")
        mode = "normal"

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1"):
                mode = "normal"
            elif key == ord("2"):
                mode = "mask"
            elif key == ord("3"):
                mode = "person_only"

            if mode == "normal":
                annotated = draw_normal(frame, result)
                cv2.imshow("Imagen - Normal", annotated)
            elif mode == "mask":
                masked = draw_mask(frame, boxes)
                cv2.imshow("Imagen - Máscara", masked)
            elif mode == "person_only":
                person_view = draw_person_only(frame, boxes)
                cv2.imshow("Imagen - Solo personas", person_view)

        cv2.destroyAllWindows()

    else:
        cap = get_video_capture()
        if cap is None:
            print("Error al abrir la fuente de video.")
            return

        mode = "normal"
        print("Presiona 1: normal | 2: máscara verde | 3: solo personas | q: salir")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = detector.detect(frame)
            boxes = result.boxes

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
                annotated = draw_normal(frame, result)
                cv2.imshow("Video - Detección normal", annotated)
            elif mode == "mask":
                masked = draw_mask(frame, boxes)
                cv2.imshow("Video - Máscara verde", masked)
            elif mode == "person_only":
                person_view = draw_person_only(frame, boxes)
                cv2.imshow("Video - Solo personas", person_view)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()