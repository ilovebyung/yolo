import cv2
import numpy as np
from ultralytics import RTDETR
from collections import defaultdict

model = RTDETR("rtdetr-l.pt")

file = 'D02.mp4'
cap = cv2.VideoCapture(file)

# Store the track history
track_history = defaultdict(lambda: [])


while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if ret:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        result = model.track(frame, persist=True)[0]

        # Get the boxes and track IDs
        if result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()
            cls = result.boxes.cls.int().cpu().tolist()

            # Visualize the result on the frame
            frame = result.plot()

            # Plot the tracks
            if cls == 0:
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
