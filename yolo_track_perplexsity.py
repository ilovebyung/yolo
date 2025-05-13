import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Initialize YOLO model and SAHI detection model
model = YOLO("yolo11n.pt")
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo11n.pt",
    confidence_threshold=0.3,
    device="cpu"  # or 'cuda:0'
)

video_path = 'D02.mp4'
cap = cv2.VideoCapture(video_path)
track_history = defaultdict(lambda: [])

while True:
    ret, frame = cap.read()
    # if not ret:
    #     break

    # Perform sliced inference with SAHI
    result = get_sliced_prediction(
        frame,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # Process detections and tracking
    if result.object_prediction_list:
        # Convert SAHI results to YOLO format
        boxes = []

        for obj in result.object_prediction_list:
            bbox = obj.bbox.to_xywh()  # Returns [x_center, y_center, width, height]

            # Ensure coordinates are absolute values
            bbox = [abs(val) for val in obj.bbox.to_xywh()]
            bbox_xyxy = obj.bbox.to_xyxy()  # [x_min, y_min, x_max, y_max]

            boxes.append([
                bbox[0],  # x-center
                bbox[1],  # y-center
                bbox[2],  # width
                bbox[3]   # height
            ])

    # if not result.object_prediction_list:
    #     continue    


        # Update track history and visualize
        if boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                history = track_history[track_id]
                history.append((float(x), float(y)))
                
                if len(history) > 30:
                    history.pop(0)
                
                # Draw tracking path
                points = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], False, (230, 230, 230), 10)

        # Display results
        frame = tracks.plot()
    
    cv2.imshow("YOLO11 SAHI Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
