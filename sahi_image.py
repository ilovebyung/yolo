# pip install -U ultralytics sahi
# Import necessary libraries
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
import cv2  
import matplotlib.pyplot as plt

# 1. Choose your YOLO model
# MODEL_PATH = "yolo11s.pt" # <--- CHANGE THIS  
MODEL_PATH = "yolov8s.pt" # <--- CHANGE THIS  

# 2. Specify the path to your image
IMAGE_PATH = './images/walking.jpg' # <--- CHANGE THIS

# 3. SAHI Slice Parameters (Adjust as needed)
SLICE_HEIGHT = 320 
SLICE_WIDTH = 320 
OVERLAP_HEIGHT_RATIO = 0.2 # Overlap between slices (20%)
OVERLAP_WIDTH_RATIO = 0.2  # Overlap between slices (20%)

# 4. Detection Confidence Threshold
CONFIDENCE_THRESHOLD = 0.5

# 5. Device ('cpu' or 'cuda:0' etc.)
DEVICE = 'cuda:0' # Change from 'cpu' to 'cuda:0' if you have a GPU and compatible PyTorch installed

# --- Setup ---
try:
    yolo_model = YOLO(MODEL_PATH)
    print(f"YOLO model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model '{MODEL_PATH}': {e}")
    print("Please ensure the model path is correct or the model name is valid.")
    exit()

# --- Load SAHI Detection Model ---
model = AutoDetectionModel.from_pretrained(
    model_type='yolo11',             # Specify the model type
    model_path=MODEL_PATH,           # Path to the YOLO model weights
    confidence_threshold=CONFIDENCE_THRESHOLD,
    device=DEVICE,                   # Set device ('cpu' or 'cuda:0')
)

image = cv2.imread(IMAGE_PATH)

sahi_result = get_sliced_prediction(
    image=image,               # Path to the image file
    detection_model=model,
    slice_height=SLICE_HEIGHT,
    slice_width=SLICE_WIDTH,
    overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
    overlap_width_ratio=OVERLAP_WIDTH_RATIO
)

# --- Process Results ---
print(f"Number of detected objects: {len(sahi_result.object_prediction_list)}")

# Draw boxes manually using OpenCV  


for prediction in sahi_result.object_prediction_list:
    bbox = prediction.bbox
    x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
    label = f"{prediction.category.name}: {prediction.score.value:.2f}"

    # Draw red bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Put label
    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (0, 0, 255), -1) # Filled background for label
    cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Black text

plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite('sahi_output.jpg',image)