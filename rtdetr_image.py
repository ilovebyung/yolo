from ultralytics import RTDETR
import cv2  
import matplotlib.pyplot as plt

# 1. Choose your YOLO model
model = RTDETR("rtdetr-l.pt") # <--- CHANGE THIS  

# 2. Specify the path to your image
image = cv2.imread('./images/walking.jpg')
results = model(image)   

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

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.imwrite('rtdetr_output.jpg',image)