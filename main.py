# Import necessary libraries
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models
# Example models: YOLOv5 for object detection and DeepLabV3 for semantic segmentation
object_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        # Save the image
        image_path = os.path.join("uploads", file.filename)
        file.save(image_path)

        # Process the image
        detections = detect_objects(image_path)

        # Return results
        return jsonify(detections)

def detect_objects(image_path):
    # Read and preprocess the image
    image = Image.open(image_path)
    results = object_detection_model(image_path)

    # Parse results
    detections = []
    for result in results.xyxy[0]:
        detections.append({
            "object": result[-1].item(),  # Class label
            "confidence": result[-2].item(),  # Confidence score
            "bbox": result[:4].tolist()  # Bounding box
        })

    return detections

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
