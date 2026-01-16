from ultralytics import YOLO
import shutil
import os

# Define paths
SOURCE_WEIGHTS = "runs/detect/train16/weights/best.pt" # Trying train16 first as I started it recently
BACKUP_WEIGHTS = "runs/detect/train15/weights/best.pt"
DESTINATION_ONNX = "models/best.onnx"

# Check if train16 exists (it might be running or finished partial)
if os.path.exists(SOURCE_WEIGHTS):
    model_path = SOURCE_WEIGHTS
    print(f"Using latest training weights: {model_path}")
elif os.path.exists(BACKUP_WEIGHTS):
    model_path = BACKUP_WEIGHTS
    print(f"Using previous training weights: {model_path}")
else:
    # Fallback to base model if no training found (unlikely)
    model_path = "yolov8m.pt"
    print(f"Using base model: {model_path}")

# Load model
model = YOLO(model_path)

# Export with Opset 12 (Stable)
print("Exporting model to ONNX with opset=12...")
success = model.export(format='onnx', opset=12)

if success:
    exported_path = model_path.replace(".pt", ".onnx")
    print(f"Export successful: {exported_path}")
    
    # Move to models folder
    shutil.move(exported_path, DESTINATION_ONNX)
    print(f"Model moved to {DESTINATION_ONNX}")
else:
    print("Export failed!")
