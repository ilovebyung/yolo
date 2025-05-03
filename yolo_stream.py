import cv2
from ultralytics import YOLO
from flask import Flask, Response

# Load the YOLOv8n model    
model = YOLO("yolo11s.pt")


app = Flask(__name__)
file = "./images/short.mp4"
camera = cv2.VideoCapture(file)  # Or your video source

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                conf = box.conf[0]
                cls = int(box.cls[0])
                if cls == 0: # persons
                
                    # Draw red bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Optionally, you can add label and confidence
                    label = f'{model.names[cls]} {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
