from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model safely
MODEL_PATH = "models/best.onnx"
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}. API will return empty results.")
except Exception as e:
    print("MODEL LOAD ERROR:", e)
    model = None


@app.get("/")
def home():
    return {"message": "PCB Defect Detection API is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model
    try:
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Invalid image"}

        if model is None:
             # Try to reload if it wasn't there before
            if os.path.exists(MODEL_PATH):
                try:
                    model = YOLO(MODEL_PATH)
                except:
                    return {"error": "Model not loaded and failed to load."}
            else:
                return {"error": "Model not loaded. Please train the model first."}

        results = model(image)

        if len(results) == 0:
            return {"boxes": [], "classes": [], "scores": []}

        result = results[0]

        # If no boxes found
        if result.boxes is None or len(result.boxes) == 0:
            return {"boxes": [], "classes": [], "scores": []}

        boxes = result.boxes.xyxy.tolist()
        classes = result.boxes.cls.tolist()
        scores = result.boxes.conf.tolist()

        return {
            "boxes": boxes,
            "classes": classes,
            "scores": scores
        }

    except Exception as e:
        return {"error": str(e)}
