# pip install -U ultralytics sahi
# Import necessary libraries
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
# No need to import YOLO separately if using AutoDetectionModel for prediction
# from ultralytics import YOLO # Keep if you need YOLO for other things, but not for SAHI prediction here
import cv2
import matplotlib.pyplot as plt
import time # To estimate processing time
import os # To check file paths

# --- Configuration ---

# 1. Choose your YOLO model (ensure it's compatible with SAHI's expected type)
#    Make sure the model file exists at this path.
MODEL_PATH = "yolo11s.pt" # <<<--- Confirm this model exists or change path/name

# 2. Specify the path to your input and output videos
VIDEO_PATH_IN = './images/short.mp4'    # <--- Input Video Path
VIDEO_PATH_OUT = 'result.mp4'   # <--- Output Video Path

# 3. SAHI Slice Parameters (Adjust as needed based on object size and video resolution)
SLICE_HEIGHT = 320
SLICE_WIDTH = 320
OVERLAP_HEIGHT_RATIO = 0.2 # Overlap between slices (20%)
OVERLAP_WIDTH_RATIO = 0.2  # Overlap between slices (20%)

# 4. Detection Confidence Threshold
CONFIDENCE_THRESHOLD = 0.7 # Lower this if objects are missed, raise if too many false positives

# 5. Device ('cpu' or 'cuda:0' etc.)
DEVICE = 'cuda:0' # Change to 'cpu' if you don't have a compatible GPU/CUDA setup

# 6. Model Type for AutoDetectionModel
MODEL_TYPE = 'yolo11' # <<<--- Make sure this matches your MODEL_PATH version

# --- Setup ---

# Check if input video exists
if not os.path.exists(VIDEO_PATH_IN):
    print(f"Error: Input video not found at '{VIDEO_PATH_IN}'")
    exit()

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    exit()

print("Loading model...")
try:
    detection_model = AutoDetectionModel.from_pretrained(
        model_type=MODEL_TYPE,
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=DEVICE,
    )
    print(f"SAHI detection model ('{MODEL_TYPE}' type, path: '{MODEL_PATH}') loaded successfully onto {DEVICE}.")
except Exception as e:
    print(f"Error loading SAHI detection model:")
    print(e)
    print("\nPlease ensure:")
    print(f"  - Model path '{MODEL_PATH}' is correct.")
    print(f"  - Model type '{MODEL_TYPE}' matches the model architecture.")
    print(f"  - Necessary dependencies (like ultralytics for YOLOv8) are installed.")
    print(f"  - If using CUDA, PyTorch and CUDA versions are compatible.")
    exit()

# --- Video Processing ---

# Open input video
cap = cv2.VideoCapture(VIDEO_PATH_IN)
if not cap.isOpened():
    print(f"Error: Could not open input video file: {VIDEO_PATH_IN}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nInput Video Properties:")
print(f"  - Resolution: {frame_width}x{frame_height}")
print(f"  - FPS: {fps:.2f}")
print(f"  - Total Frames: {total_frames if total_frames > 0 else 'Unknown (stream?)'}")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_PATH_OUT, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print(f"Error: Could not open output video file for writing: {VIDEO_PATH_OUT}")
    cap.release()
    exit()

print(f"\nProcessing video: '{VIDEO_PATH_IN}'...")
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Reached end of video or error reading frame.")
        break

    sahi_result = get_sliced_prediction(
        image=frame,                 # Pass the BGR frame directly
        detection_model=detection_model, # Use the preloaded model
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
        # verbose=0 # Set verbose level (0: no messages, 1: basic, 2: detailed) - can slow down processing
    )

    # --- Draw Bounding Boxes on the frame ---
    object_prediction_list = sahi_result.object_prediction_list

    for prediction in object_prediction_list:
        bbox = prediction.bbox
        # Ensure coordinates are integers for OpenCV drawing functions
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        label = f"{prediction.category.name}: {prediction.score.value:.2f}"

        # Draw red bounding box (BGR format, so (0, 0, 255) is red)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red box, thickness 2

        # Prepare label text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw filled rectangle background for label
        # Adjust position slightly above the box, handling boundary cases
        label_y = max(y1 - baseline, label_height + baseline) # Ensure label isn't drawn off-screen top
        cv2.rectangle(frame, (x1, label_y - label_height - baseline), (x1 + label_width, label_y), (0, 0, 255), -1) # Filled red background

        # Put label text (white color on red background)
        cv2.putText(frame, label, (x1, label_y - baseline // 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA) # White text

    # --- Write the processed frame to the output video ---
    out.write(frame)

    if frame_count % 10 == 0:  # Display every 10th frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.show()

    frame_count += 1  # Increment frame counter

    # --- Print progress (optional, can slow down processing slightly) ---
    if frame_count % 30 == 0: # Print progress every 30 frames
       elapsed_time = time.time() - start_time
       eta = 'N/A'
       if total_frames > 0 and frame_count > 0:
           fps_proc = frame_count / elapsed_time
           remaining_frames = total_frames - frame_count
           eta_seconds = remaining_frames / fps_proc if fps_proc > 0 else 0
           eta = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

       print(f"  Processed Frame: {frame_count}/{total_frames if total_frames > 0 else '?'}"
             f" | Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
             f" | ETA: {eta}")


# --- Cleanup ---
print("\nFinishing up...")
cap.release()   # Release the input video capture object
out.release()   # Release the output video writer object

end_time = time.time()
total_processing_time = end_time - start_time
avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0

print(f"\n--- Processing Complete ---")
print(f"Output video saved to: '{VIDEO_PATH_OUT}'")
print(f"Total frames processed: {frame_count}")
print(f"Total processing time: {time.strftime('%H:%M:%S', time.gmtime(total_processing_time))}")
print(f"Average processing FPS: {avg_fps:.2f}")