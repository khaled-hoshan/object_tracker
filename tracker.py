import cv2
from ultralytics import YOLO  # type: ignore
from sort import TrackerManager
import numpy as np

model = YOLO("yolov8n.pt")
tracker = TrackerManager(max_age=15, min_hits=10, iou_threshold=0.3)

cap = cv2.VideoCapture("test_video.mp4")

# CRITICAL PARAMETERS
CONFIDENCE_THRESHOLD = 0.37
NMS_IOU_THRESHOLD = 0.6  # Remove duplicate detections

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect with NMS enabled
    results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=NMS_IOU_THRESHOLD)[0]

    detections = []
    for box in results.boxes.xyxy:
        detections.append(box.cpu().numpy())

    # Track
    tracked = tracker.update(detections)

    # Draw
    for obj in tracked:
        x1, y1, x2, y2, track_id = obj
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID:{int(track_id)}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
