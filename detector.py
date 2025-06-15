
from ultralytics import YOLO
from modelo import MODEL_PATH, CONFIDENCE_THRESHOLD

class YoloDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)

    def detect(self, frame):
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD)
        return results[0]
