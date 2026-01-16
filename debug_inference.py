from ultralytics import YOLO
import cv2

# Load model
model = YOLO("models/best.onnx", task="detect")

# Load image
img_path = "data/pcb-small/train/images/burnt_02.jpg"

# Predict
results = model(img_path)

# Print results
import numpy as np
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Mean Brightness:", np.mean(gray))
print("Std Dev:", np.std(gray))

for r in results:
    print("Boxes:", r.boxes.xyxy.tolist())
    print("Classes:", r.boxes.cls.tolist())
    print("Scores:", r.boxes.conf.tolist())
