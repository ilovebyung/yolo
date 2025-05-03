import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Initialize the webcam
cap = cv2.VideoCapture(0)
speeds = []

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model(frame)
    speeds.append(results[0].speed['inference'])

    # Visualize the results on the frame
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            # Draw red bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Optionally, you can add label and confidence
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("YOLO 11 Inference", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # import matplotlib.pyplot as plt
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

print(sum(speeds)/len(speeds))
print("YOLO 11 Inference", sum(speeds)/len(speeds))

