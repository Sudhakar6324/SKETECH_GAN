import cv2

# Load a face photo
img = cv2.imread("IMG_0286.JPG")

# Convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Invert
inv = 255 - gray

# Blur
blur = cv2.GaussianBlur(inv, (21,21), 0)

# Dodge blend
sketch = cv2.divide(gray, 255 - blur, scale=256)

cv2.imwrite("face_sketch.jpg", sketch)
