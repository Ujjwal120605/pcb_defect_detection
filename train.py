from ultralytics import YOLO
import shutil
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

model = YOLO('yolov8m.pt')
# Train the model
results = model.train(data='data_small.yaml', epochs=100, imgsz=640)

# Export the model to ONNX
success = model.export(format='onnx')

# The exported model is usually saved in runs/detect/train/weights/best.onnx or similar.
# Ultralytics export usually returns the path to the exported file.
# However, to be safe and consistent, let's find it or use the return value if possible.
# Simpler approach: The export command saves it in the same dir as the model pt file usually,
# or we can just look for it.
# Actually, model.export() returns the filename.

# Let's try to locate the best.pt and export that specifically if we want the best trained model.
# The 'model' object after training should be the trained one.

# Let's explicitly load the best trained model to export
best_model_path = str(results.save_dir / 'weights' / 'best.pt')
print(f"Loading best model from: {best_model_path}")
best_model = YOLO(best_model_path)
exported_path = best_model.export(format='onnx')

# Move to models/best.onnx
target_path = 'models/best.onnx'
if isinstance(exported_path, str):
    shutil.move(exported_path, target_path)
    print(f"Model saved to {target_path}")
else:
    print(f"Export returned {exported_path}, check where the file is.")
