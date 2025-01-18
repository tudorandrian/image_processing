# Import necessary libraries
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
import torch
from torchvision import transforms
from PIL import Image
import os
import uuid

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
    # Render the home page with the upload form
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if the image and action are in the request
    if 'image' not in request.files or 'action' not in request.form:
        return jsonify({"error": "No file part or action selected"})
    file = request.files['image']
    action = request.form['action']

    # Check if a file is selected
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        # Generate a unique filename if the file already exists
        filename = file.filename
        image_path = os.path.join("uploads", filename)
        if os.path.exists(image_path):
            filename = f"{uuid.uuid4().hex}_{filename}"
            image_path = os.path.join("uploads", filename)

        # Save the image to the uploads directory
        file.save(image_path)

        # Redirect to the results page with the filename and action as query parameters
        return redirect(url_for('results', filename=filename, action=action))

@app.route('/results')
def results():
    # Get the filename and action from the query parameters
    filename = request.args.get('filename')
    action = request.args.get('action')
    image_path = os.path.join("uploads", filename)

    # Process the image based on the selected action
    processed_image_path = process_image(image_path, action)

    # Render the results page with the uploaded image and selected action
    return render_template('results.html', filename=filename, action=action, processed_image=processed_image_path)

def process_image(image_path, action):
    # Read and preprocess the image
    image = Image.open(image_path)
    results = object_detection_model(image_path)

    # Draw bounding boxes on the detected objects
    image_np = np.array(image)
    for result in results.xyxy[0]:
        bbox = result[:4].tolist()
        cv2.rectangle(image_np, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

    # Save the processed image
    processed_image_filename = f"processed_{os.path.basename(image_path)}"
    processed_image_path = os.path.join("uploads", processed_image_filename)
    Image.fromarray(image_np).save(processed_image_path)

    return processed_image_filename

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Serve the uploaded file from the uploads directory
    return send_from_directory('uploads', filename)

if __name__ == "__main__":
    # Create the uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    # Run the Flask app in debug mode
    app.run(debug=True)