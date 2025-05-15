from ultralytics import YOLO
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# 1. Choose your YOLO model
model = YOLO('yolov8s.pt') # <--- CHANGE THIS  

# 2. Specify the path to your image
image = cv2.imread('./images/walking.jpg')
# results = model(source=image)  
results = model(source=image,imgsz=(1381, 2069),conf=0.5)  

# --- Process Results ---
print(f"Number of detected objects: {len(results[0])}")

for result in results:
    boxes = result.boxes.cpu().numpy()
    for box in boxes:
        # Get the coordinates
        x1, y1, x2, y2 = box.xyxy[0].astype(int)

        # Get the class and confidence
        class_id = box.cls[0].astype(int)
        conf = box.conf[0]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Put class name and confidence
        label = f'{model.names[class_id]} {conf:.2f}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # # Detection
    # result.boxes.xyxy   # box with xyxy format, (N, 4)
    # result.boxes.xywh   # box with xywh format, (N, 4)
    # result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    # result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
    # result.boxes.conf   # confidence score, (N, 1)
    # result.boxes.cls    # cls, (N, 1)

    # # Segmentation
    # result.masks.data      # masks, (N, H, W)
    # result.masks.xy        # x,y segments (pixels), List[segment] * N
    # result.masks.xyn       # x,y segments (normalized), List[segment] * N

    # # Classification
    # prob = result.probs     # cls prob, (num_class, )

    # # Filter the results with a confidence score greater than 0.5
    # filtered_results = result.boxes[prob > 0.5]

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.imwrite('yolo_output.jpg',image)