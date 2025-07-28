import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create MOG2 background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply background subtraction 
    fgmask = fgbg.apply(gray)
    # Apply morphological operations to reduce noise
    fgmask_cleaned = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # Bitwise AND to isolate moving pixels in grayscale
    moving_pixels_gray = cv2.bitwise_and(gray, fgmask_cleaned)

    # Display result
    cv2.imshow("Grayscale Moving Pixels (MOG2)", moving_pixels_gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
