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
# YOLOv5 for object detection
object_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# DeepLabV3 for image segmentation
segmentation_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
segmentation_model.eval()

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
    color_space = request.form['color_space']

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
        return redirect(url_for('results', filename=filename, action=action, color_space=color_space))

@app.route('/results')
def results():
    # Get the filename and action from the query parameters
    filename = request.args.get('filename')
    action = request.args.get('action')
    color_space = request.args.get('color_space')
    image_path = os.path.join("uploads", filename)

    # Process the image based on the selected action
    processed_image_path = process_image(image_path, action)
    segmented_image_path = segment_image(image_path, action)
    # Convert the image to the selected color space
    converted_image_path = convert_color_space(image_path, color_space)

    # Render the results page with the uploaded image, processed image, and segmented image
    return render_template('results.html', filename=filename, action=action, color_space=color_space, processed_image=processed_image_path, segmented_image=segmented_image_path, converted_image=converted_image_path)

def process_image(image_path, action):
    # Read and preprocess the image
    image = Image.open(image_path)
    results = object_detection_model(image_path)

    # Draw bounding boxes on the detected objects
    image_np = np.array(image)
    for result in results.xyxy[0]:
        bbox = result[:4].tolist()
        label = result[-1].item()  # Get the class label
        confidence = result[-2].item()  # Get the confidence score

        # Customize properties
        color = (0, 255, 0)  # Green color
        thickness = 3  # Line thickness
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        # Draw the bounding box
        cv2.rectangle(image_np, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, thickness)

        # Add label and confidence score
        label_text = f"{label}: {confidence:.2f}"
        cv2.putText(image_np, label_text, (int(bbox[0]), int(bbox[1]) - 10), font, font_scale, color, font_thickness)

    # Save the processed image
    processed_image_filename = f"processed_{os.path.basename(image_path)}"
    processed_image_path = os.path.join("uploads", processed_image_filename)
    Image.fromarray(image_np).save(processed_image_path)

    return processed_image_filename

def segment_image(image_path, action):
    # Read and preprocess the image
    image = Image.open(image_path)
    image_tensor = preprocess(image).unsqueeze(0)

    # Perform segmentation
    with torch.no_grad():
        output = segmentation_model(image_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

    # Convert the segmentation map to an image
    segmented_image = Image.fromarray(output_predictions)
    segmented_image = segmented_image.resize(image.size)

    # Convert to numpy array for OpenCV processing
    segmented_image_np = np.array(segmented_image)

    # Customize properties
    color_map = {
        0: (0, 0, 0),  # Background
        1: (0, 255, 0),  # Class 1
        2: (0, 0, 255),  # Class 2
        # Add more classes as needed
    }
    transparency = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Create an overlay for transparency
    overlay = np.array(image).copy()
    for class_id, color in color_map.items():
        mask = (segmented_image_np == class_id)
        overlay[mask] = color

    # Blend the overlay with the original image
    blended_image = cv2.addWeighted(np.array(image), 1 - transparency, overlay, transparency, 0)

    # Draw contours around segmented regions
    for class_id, color in color_map.items():
        mask = (segmented_image_np == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended_image, contours, -1, color, 2)

    # Add labels to the segmented regions
    for class_id, color in color_map.items():
        mask = (segmented_image_np == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            label_text = f"Class {class_id}"
            cv2.putText(blended_image, label_text, (x, y - 10), font, font_scale, color, font_thickness)

    # Save the segmented image
    segmented_image_filename = f"segmented_{os.path.basename(image_path)}"
    segmented_image_path = os.path.join("uploads", segmented_image_filename)
    Image.fromarray(blended_image).save(segmented_image_path)

    return segmented_image_filename

def convert_color_space(image_path, color_space):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to the selected color space
    if color_space == 'HSV':
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == 'LAB':
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif color_space == 'GRAY':
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        converted_image = image

    # Save the converted image
    converted_image_filename = f"converted_{os.path.basename(image_path)}"
    converted_image_path = os.path.join("uploads", converted_image_filename)
    cv2.imwrite(converted_image_path, converted_image)

    # Log the conversion process
    print(f"Converted image saved at: {converted_image_path} with color space: {color_space}")

    return converted_image_filename

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