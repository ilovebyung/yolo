import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt


# Initialize YOLO and background subtractor
model = YOLO("yolo11s.pt")
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16)

# Video capture
# Initialize video capture
file = './images/short.mp4'
cap = cv2.VideoCapture(file)   

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Background subtraction
    fg_mask = bg_subtractor.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # YOLO detection
    results = model(frame)

    '''
    Filter Detections with Foreground Mask: Check if YOLO's bounding boxes overlap with the foreground mask. 
    Only keep detections where the bounding box contains a significant portion of the foreground (moving) pixels.
    For each bounding box, compute the intersection with the mask.
    If the overlap (e.g., percentage of masked pixels inside the box) exceeds a threshold (e.g., 30%), consider it a moving object.
    '''
    # Visualize the results on the frame
    for result in results[0].boxes:
        cls = int(result.cls[0])
        if cls == 0: # persons
            conf = result.conf[0]
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            roi = fg_mask[y1:y2, x1:x2]
            if cv2.countNonZero(roi) / roi.size > 0.3:  # Threshold for motion
                # Keep this detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Optionally, you can add label and confidence
                label = f'{model.names[cls]} {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
   
    if frame_count % 10 == 0:  # Display every 10th frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.show()
        fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2RGB)
        plt.imshow(fg_mask)
        plt.show()

    frame_count += 1  # Increment frame counter

cap.release()
