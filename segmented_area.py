from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

file = './images/sample.jpg'

# get a sample of uint8
image = cv2.imread(file)
plt.imshow(image, cmap='gray')
h, w, c = image.shape

# make a prediction
model = YOLO('best.pt')
results = model.predict(source=image, conf=0.8)  # save plotted images

for result in results:
    # Segmentation
    data = result.masks.data      # masks, (N, H, W)
    xy = result.masks.xy        # x,y segments (pixels), List[segment] * N
    xyn = result.masks.xyn       # x,y segments (normalized), List[segment] * N


# generate mask
mask = data[0]
# Convert the tensor to a NumPy array
mask = mask.cpu().numpy()*255

# resize image
mask = cv2.resize(mask, (w,h), interpolation = cv2.INTER_AREA)
mask = mask.astype(np.uint8)
plt.imshow(mask, cmap='gray')

# Apply the mask to the image
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
masked_image = cv2.bitwise_and(image, mask)
cv2.imwrite('masked_image.jpg', masked_image)
plt.imshow(masked_image, cmap='gray')

# Calculate area  
area = np.count_nonzero(mask == 0)
print(area)

# Calculate brightness excluding masked area 
brightness = np.median(masked_image[masked_image != 0]).astype('uint8')
print(brightness)

