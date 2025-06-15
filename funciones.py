# modes.py

import cv2
import numpy as np

def draw_normal(frame, results):
    return results.plot()

def draw_mask(frame, boxes):
    overlay = frame.copy()
    alpha = 0.5
    for box in boxes:
        if int(box.cls[0]) == 0:  # Persona
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def draw_person_only(frame, boxes):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for box in boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    people = cv2.bitwise_and(frame, frame, mask=mask)
    bg = np.full_like(frame, (50, 50, 50))
    inv_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(bg, bg, mask=inv_mask)
    return cv2.add(people, background)