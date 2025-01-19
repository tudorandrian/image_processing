import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory, flash
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
from deepface import DeepFace

app = Flask(__name__)

# Load pre-trained models
# object_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
segmentation_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
segmentation_model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# List of COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No file part or action selected"})
    file = request.files['image']
    result_option = request.form['result_option']  # Capture the selected result option
    color_space = request.form['color_space']
    rotation_angle = request.form.get('rotation_angle', type=int, default=0)
    scale_factor = request.form.get('scale_factor', type=float, default=1.0)
    crop_x = request.form.get('crop_x', type=int, default=0)
    crop_y = request.form.get('crop_y', type=int, default=0)
    crop_width = request.form.get('crop_width', type=int, default=100)
    crop_height = request.form.get('crop_height', type=int, default=100)
    flip_code = request.form.get('flip_code', type=int, default=1)
    filter_type = request.form.get('filter_type')
    kernel_size = request.form.get('kernel_size', type=int, default=7)
    edge_algorithm = request.form.get('edge_algorithm')
    threshold1 = request.form.get('threshold1', type=int, default=100)
    threshold2 = request.form.get('threshold2', type=int, default=200)
    equalization_type = request.form.get('equalization_type')
    clip_limit = request.form.get('clip_limit', type=float, default=2.0)
    tile_grid_size = request.form.get('tile_grid_size', type=int, default=8)
    enhancement_type = request.form.get('enhancement_type')
    enhancement_value = request.form.get('enhancement_value', type=float, default=1.0)
    # Capture the selected model from the form data
    model_name = request.form['model']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        filename = file.filename
        image_path = os.path.join("uploads", filename)
        if os.path.exists(image_path):
            filename = f"{uuid.uuid4().hex}_{filename}"
            image_path = os.path.join("uploads", filename)
        file.save(image_path)

        return redirect(url_for('results', filename=filename, result_option=result_option, color_space=color_space,
                                rotation_angle=rotation_angle, scale_factor=scale_factor,
                                crop_x=crop_x, crop_y=crop_y, crop_width=crop_width, crop_height=crop_height,
                                flip_code=flip_code, filter_type=filter_type, kernel_size=kernel_size,
                                edge_algorithm=edge_algorithm, threshold1=threshold1, threshold2=threshold2,
                                equalization_type=equalization_type, clip_limit=clip_limit, tile_grid_size=tile_grid_size,
                                enhancement_type=enhancement_type, enhancement_value=enhancement_value, model_name=model_name))

def detect_emotions(image_path):
    """
    Detect emotions on faces in the given image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        list: A list of dictionaries containing detected emotions and their probabilities.
    """
    # Analyze the image to detect emotions
    results = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)

    return results

@app.route('/results')
def results():
    try:
        filename = request.args.get('filename')
        result_option = request.args.get('result_option')
        color_space = request.args.get('color_space')
        rotation_angle = request.args.get('rotation_angle', type=int)
        scale_factor = request.args.get('scale_factor', type=float)
        crop_x = request.args.get('crop_x', type=int)
        crop_y = request.args.get('crop_y', type=int)
        crop_width = request.args.get('crop_width', type=int)
        crop_height = request.args.get('crop_height', type=int)
        flip_code = request.args.get('flip_code', type=int)
        filter_type = request.args.get('filter_type')
        kernel_size = request.args.get('kernel_size', type=int, default=7)
        edge_algorithm = request.args.get('edge_algorithm')
        threshold1 = request.args.get('threshold1', type=int)
        threshold2 = request.args.get('threshold2', type=int)
        equalization_type = request.args.get('equalization_type')
        clip_limit = request.args.get('clip_limit', type=float)
        tile_grid_size = request.args.get('tile_grid_size', type=int)
        enhancement_type = request.args.get('enhancement_type')
        enhancement_value = request.args.get('enhancement_value', type=float)
        image_path = os.path.join("uploads", filename)

        # Get the selected model name from the request arguments
        model_name = request.args.get('model_name')

        # Load the selected object detection model
        object_detection_model = torch.hub.load('ultralytics/yolov5', model_name)

        processed_image_path, detected_classes, processed_image_person_path = process_image(image_path, object_detection_model)
        segmented_image_path, segmentation_metrics = segment_image(image_path)

        converted_image_paths = convert_color_space(image_path, color_space, result_option)
        transformed_image_path = transform_image(image_path, rotation_angle, scale_factor, crop_x, crop_y, crop_width, crop_height, flip_code)
        filtered_image_paths = apply_filter(image_path, filter_type, kernel_size, result_option)
        edge_detected_image_paths = apply_edge_detection(image_path, edge_algorithm, threshold1, threshold2, result_option)
        equalized_image_paths = apply_histogram_equalization(image_path, equalization_type, clip_limit, tile_grid_size, result_option)
        enhanced_image_paths = apply_image_enhancement(image_path, enhancement_type, enhancement_value, result_option)

        # Ensure converted_image_paths has enough elements
        converted_images = {}
        color_spaces = ['RGB', 'HSV', 'LAB', 'GRAY']
        for i, cs in enumerate(color_spaces):
            if i < len(converted_image_paths):
                converted_images[cs] = converted_image_paths[i]

        # Dictionary of color spaces with descriptions
        color_space_descriptions = {
            'RGB': 'RGB color space.',
            'HSV': 'Hue, Saturation, and Value color space.',
            'LAB': 'CIE L*a*b* color space.',
            'GRAY': 'Grayscale color space.'
        }

        # Ensure filtered_image_paths has enough elements
        filtered_images = {}
        filter_types = ['gaussian', 'median']
        for i, ft in enumerate(filter_types):
            if i < len(filtered_image_paths):
                filtered_images[ft] = filtered_image_paths[i]

        # Dictionary of filter types with descriptions
        filter_type_descriptions = {
            'gaussian': 'Gaussian Blur filter.',
            'median': 'Median Blur filter.'
        }

        # Ensure edge_detected_image_paths has enough elements
        edge_detected_images = {}
        algorithms = ['canny', 'sobel', 'scharr', 'roberts', 'log']
        for i, algorithm in enumerate(algorithms):
            if i < len(edge_detected_image_paths):
                edge_detected_images[algorithm] = edge_detected_image_paths[i]

        # Dictionary of edge detection algorithms with descriptions
        edge_algorithm_descriptions = {
            'canny': 'Canny edge detection algorithm.',
            'sobel': 'Sobel edge detection algorithm.',
            'scharr': 'Scharr edge detection algorithm.',
            'roberts': 'Roberts edge detection algorithm.',
            'log': 'Laplacian of Gaussian (LoG) edge detection algorithm.'
        }

        # Ensure equalized_image_paths has enough elements
        equalized_images = {}
        equalization_types = ['ahe', 'clahe']
        for i, et in enumerate(equalization_types):
            if i < len(equalized_image_paths):
                equalized_images[et] = equalized_image_paths[i]

        # Dictionary of equalization types with descriptions
        equalization_type_descriptions = {
            'ahe': 'Adaptive Histogram Equalization.',
            'clahe': 'Contrast Limited Adaptive Histogram Equalization.'
        }

        # Check if a person is detected and perform emotion detection
        emotions = []
        if 'person' in detected_classes:
            emotions = detect_emotions(image_path)

        return render_template('results.html', filename=filename, result_option=result_option,
                               model_name=model_name, processed_image=processed_image_path,
                               processed_image_person=processed_image_person_path, emotions=emotions,
                               segmented_image=segmented_image_path,
                               color_space=color_space, converted_images=converted_images,
                               color_space_descriptions=color_space_descriptions,
                               transformed_image=transformed_image_path,
                               filtered_images=filtered_images, filter_type_descriptions=filter_type_descriptions,
                               equalized_images=equalized_images,
                               equalization_type_descriptions=equalization_type_descriptions,
                               enhanced_images=enhanced_image_paths,
                               detected_classes=detected_classes,
                               segmentation_metrics=segmentation_metrics,
                               edge_algorithm=edge_algorithm, edge_detected_images=edge_detected_images,
                               edge_algorithm_descriptions=edge_algorithm_descriptions)

    except ValueError as e:
        flash(str(e))
        return redirect(url_for('index'))

def process_image(image_path, object_detection_model):
    image = Image.open(image_path)
    results = object_detection_model(image_path)
    image_np = np.array(image)

    # List to store detected class names
    detected_classes = []

    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for result in results.xyxy[0]:
        bbox = result[:4].tolist()
        class_id = int(result[-1].item())
        confidence = result[-2].item()
        color = (0, 255, 0)
        thickness = 3
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        cv2.rectangle(image_np, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, thickness)
        label_text = f"{COCO_CLASSES[class_id]}: {confidence:.2f}"
        cv2.putText(image_np, label_text, (int(bbox[0]), int(bbox[1]) - 10), font, font_scale, color, font_thickness)
        detected_classes.append(COCO_CLASSES[class_id])

    processed_image_filename = f"processed_{os.path.basename(image_path)}"
    processed_image_path = os.path.join("uploads", processed_image_filename)
    Image.fromarray(image_np).save(processed_image_path)

    # Check if a person is detected
    if 'person' in detected_classes:
        # Convert the image to grayscale for face detection
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw a blue circle around each detected face
        for (x, y, w, h) in faces:
            center = (x + w // 2, y + h // 2)
            radius = w // 2
            cv2.circle(image_np, center, radius, (255, 0, 0), 3)

    processed_image_person_filename = f"processed_person_{os.path.basename(image_path)}"
    processed_image_person_path = os.path.join("uploads", processed_image_person_filename)
    Image.fromarray(image_np).save(processed_image_person_path)

    return processed_image_filename, list(set(detected_classes)), processed_image_person_filename

def segment_image(image_path):
    image = Image.open(image_path)
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = segmentation_model(image_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    segmented_image = Image.fromarray(output_predictions)
    segmented_image = segmented_image.resize(image.size)
    segmented_image_np = np.array(segmented_image)
    color_map = {0: (0, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
    transparency = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    overlay = np.array(image).copy()
    for class_id, color in color_map.items():
        mask = (segmented_image_np == class_id)
        overlay[mask] = color
    blended_image = cv2.addWeighted(np.array(image), 1 - transparency, overlay, transparency, 0)
    for class_id, color in color_map.items():
        mask = (segmented_image_np == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended_image, contours, -1, color, 2)
    for class_id, color in color_map.items():
        mask = (segmented_image_np == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            label_text = f"Class {class_id}"
            cv2.putText(blended_image, label_text, (x, y - 10), font, font_scale, color, font_thickness)

    segmented_image_filename = f"segmented_{os.path.basename(image_path)}"
    segmented_image_path = os.path.join("uploads", segmented_image_filename)
    Image.fromarray(blended_image).save(segmented_image_path)

    segmented_image_np = np.array(Image.open(segmented_image_path))
    segmentation_metrics = calculate_segmentation_metrics(segmented_image_np)

    return segmented_image_filename, segmentation_metrics

def convert_color_space(image_path, color_space, result_option):
    image = cv2.imread(image_path)
    results = []

    if result_option == 'all':
        # Return all color space conversions
        color_spaces = ['HSV', 'LAB', 'GRAY']
        for cs in color_spaces:
            if cs == 'HSV':
                converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif cs == 'LAB':
                converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            elif cs == 'GRAY':
                converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            converted_image_filename = f"converted_{cs}_{os.path.basename(image_path)}"
            converted_image_path = os.path.join("uploads", converted_image_filename)
            cv2.imwrite(converted_image_path, converted_image)
            results.append(converted_image_filename)
    else:
        # Return single color space conversion
        if color_space == 'HSV':
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == 'LAB':
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif color_space == 'GRAY':
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            converted_image = image
        converted_image_filename = f"converted_{os.path.basename(image_path)}"
        converted_image_path = os.path.join("uploads", converted_image_filename)
        cv2.imwrite(converted_image_path, converted_image)
        results.append(converted_image_filename)

    return results

def transform_image(image_path, rotation_angle, scale_factor, crop_x, crop_y, crop_width, crop_height, flip_code):
    image = cv2.imread(image_path)
    if rotation_angle != 0:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
    if scale_factor != 1.0:
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    if crop_width > 0 and crop_height > 0:
        image = image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
    if flip_code in [0, 1, -1]:
        image = cv2.flip(image, flip_code)
    transformed_image_filename = f"transformed_{os.path.basename(image_path)}"
    transformed_image_path = os.path.join("uploads", transformed_image_filename)
    cv2.imwrite(transformed_image_path, image)
    print(f"Transformed image saved at: {transformed_image_path}")
    return transformed_image_filename

def apply_filter(image_path, filter_type, kernel_size, result_option):
    image = cv2.imread(image_path)
    results = []

    if result_option == 'all':
        # Return all filter types
        filter_types = ['gaussian', 'median']
        for ft in filter_types:
            if ft == 'gaussian':
                filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            elif ft == 'median':
                filtered_image = cv2.medianBlur(image, kernel_size)
            filtered_image_filename = f"filtered_{ft}_{os.path.basename(image_path)}"
            filtered_image_path = os.path.join("uploads", filtered_image_filename)
            cv2.imwrite(filtered_image_path, filtered_image)
            results.append(filtered_image_filename)
    else:
        # Return single filter type
        if filter_type == 'gaussian':
            filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif filter_type == 'median':
            filtered_image = cv2.medianBlur(image, kernel_size)
        else:
            filtered_image = image
        filtered_image_filename = f"filtered_{os.path.basename(image_path)}"
        filtered_image_path = os.path.join("uploads", filtered_image_filename)
        cv2.imwrite(filtered_image_path, filtered_image)
        results.append(filtered_image_filename)

    return results

def apply_edge_detection(image_path, edge_algorithm, threshold1, threshold2, result_option):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    results = []

    if result_option == 'all':
        # Return all edge detection algorithms
        edge_algorithms = ['canny', 'sobel', 'scharr', 'roberts', 'log']
        for ea in edge_algorithms:
            if ea == 'canny':
                edges = cv2.Canny(image, threshold1, threshold2)
            elif ea == 'sobel':
                edges = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
            elif ea == 'scharr':
                edges = cv2.Scharr(image, cv2.CV_64F, 1, 0)
            elif ea == 'roberts':
                kernel = np.array([[1, 0], [0, -1]], dtype=np.float32)
                edges = cv2.filter2D(image, -1, kernel)
            elif ea == 'log':
                edges = cv2.Laplacian(image, cv2.CV_64F)
            edge_detected_image_filename = f"edge_detected_{ea}_{os.path.basename(image_path)}"
            edge_detected_image_path = os.path.join("uploads", edge_detected_image_filename)
            cv2.imwrite(edge_detected_image_path, edges)
            results.append(edge_detected_image_filename)
    else:
        # Return single edge detection algorithm
        if edge_algorithm == 'canny':
            edges = cv2.Canny(image, threshold1, threshold2)
        elif edge_algorithm == 'sobel':
            edges = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        elif edge_algorithm == 'scharr':
            edges = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        elif edge_algorithm == 'roberts':
            kernel = np.array([[1, 0], [0, -1]], dtype=np.float32)
            edges = cv2.filter2D(image, -1, kernel)
        elif edge_algorithm == 'log':
            edges = cv2.Laplacian(image, cv2.CV_64F)
        else:
            edges = image
        edge_detected_image_filename = f"edge_detected_{os.path.basename(image_path)}"
        edge_detected_image_path = os.path.join("uploads", edge_detected_image_filename)
        cv2.imwrite(edge_detected_image_path, edges)
        results.append(edge_detected_image_filename)

    return results

def apply_histogram_equalization(image_path, equalization_type, clip_limit, tile_grid_size, result_option):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    results = []

    if result_option == 'all':
        # Return all equalization types
        equalization_types = ['ahe', 'clahe']
        for et in equalization_types:
            if et == 'ahe':
                equalized_image = cv2.equalizeHist(image)
            elif et == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
                equalized_image = clahe.apply(image)
            equalized_image_filename = f"equalized_{et}_{os.path.basename(image_path)}"
            equalized_image_path = os.path.join("uploads", equalized_image_filename)
            cv2.imwrite(equalized_image_path, equalized_image)
            results.append(equalized_image_filename)
    else:
        # Return single equalization type
        if equalization_type == 'ahe':
            equalized_image = cv2.equalizeHist(image)
        elif equalization_type == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            equalized_image = clahe.apply(image)
        else:
            equalized_image = image
        equalized_image_filename = f"equalized_{os.path.basename(image_path)}"
        equalized_image_path = os.path.join("uploads", equalized_image_filename)
        cv2.imwrite(equalized_image_path, equalized_image)
        results.append(equalized_image_filename)

    return results

def apply_image_enhancement(image_path, enhancement_type, enhancement_value, result_option):
    image = Image.open(image_path)
    results = []

    if result_option == 'all':
        # Return all enhancement types
        enhancement_types = ['sharpen', 'denoise', 'brightness', 'contrast']
        for et in enhancement_types:
            if et == 'sharpen':
                enhancer = ImageEnhance.Sharpness(image)
            elif et == 'denoise':
                enhancer = ImageEnhance.Sharpness(image)  # Placeholder, use actual denoising method
            elif et == 'brightness':
                enhancer = ImageEnhance.Brightness(image)
            elif et == 'contrast':
                enhancer = ImageEnhance.Contrast(image)
            enhanced_image = enhancer.enhance(enhancement_value)
            enhanced_image_filename = f"enhanced_{et}_{os.path.basename(image_path)}"
            enhanced_image_path = os.path.join("uploads", enhanced_image_filename)
            enhanced_image.save(enhanced_image_path)
            results.append(enhanced_image_filename)
    else:
        # Return single enhancement type
        if enhancement_type == 'sharpen':
            enhancer = ImageEnhance.Sharpness(image)
        elif enhancement_type == 'denoise':
            enhancer = ImageEnhance.Sharpness(image)  # Placeholder, use actual denoising method
        elif enhancement_type == 'brightness':
            enhancer = ImageEnhance.Brightness(image)
        elif enhancement_type == 'contrast':
            enhancer = ImageEnhance.Contrast(image)
        else:
            enhancer = None

        if enhancer:
            enhanced_image = enhancer.enhance(enhancement_value)
        else:
            enhanced_image = image

        enhanced_image_filename = f"enhanced_{os.path.basename(image_path)}"
        enhanced_image_path = os.path.join("uploads", enhanced_image_filename)
        enhanced_image.save(enhanced_image_path)
        results.append(enhanced_image_filename)

    return results

# Function to calculate segmentation metrics
def calculate_segmentation_metrics(segmented_image_np):
    """
    Calculate segmentation metrics such as the area of each segmented region,
    the number of segments, and the percentage of the image covered by each class.

    Args:
        segmented_image_np (numpy.ndarray): The segmented image as a NumPy array.

    Returns:
        dict: A dictionary containing the segmentation metrics.
    """
    unique, counts = np.unique(segmented_image_np, return_counts=True)
    total_pixels = segmented_image_np.size
    metrics = {
        "num_segments": len(unique),
        "segments": []
    }
    for segment, count in zip(unique, counts):
        area = count
        percentage = (count / total_pixels) * 100
        metrics["segments"].append({
            "segment_id": int(segment),
            "area": area,
            "percentage": percentage
        })
    return metrics

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)