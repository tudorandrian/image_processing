import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory, flash
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
from deepface import DeepFace

# Initialize the Flask application
app = Flask(__name__)

# Load pre-trained models
segmentation_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
segmentation_model.eval()

# Define a preprocessing pipeline for image transformation
preprocess = transforms.Compose([
    # Resize the image to 224x224 pixels
    transforms.Resize((224, 224)),
    # Convert the image to a tensor
    transforms.ToTensor(),
    # Normalize the image with mean and standard deviation values
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# List of COCO class names
# The list COCO_CLASSES represents the class labels for objects that the COCO (Common Objects in Context) dataset recognizes
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

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the docs route
@app.route('/docs')
def docs():
    return render_template('docs.html')


# Define the upload route to handle image uploads and processing
@app.route('/upload', methods=['POST'])
def upload():
    # Check if the request contains an image file
    if 'image' not in request.files:
        return jsonify({"error": "No file part or action selected"})

    # Get the uploaded image file
    file = request.files['image']

    # Capture the selected result option from the form data
    result_option = request.form['result_option']
    # Capture the selected color space from the form data
    color_space = request.form['color_space']
    # Capture the rotation angle from the form data, default to 0 if not provided
    rotation_angle = request.form.get('rotation_angle', type=int, default=0)
    # Capture the scale factor from the form data, default to 1.0 if not provided
    scale_factor = request.form.get('scale_factor', type=float, default=1.0)
    # Capture the crop x-coordinate from the form data, default to 0 if not provided
    crop_x = request.form.get('crop_x', type=int, default=0)
    # Capture the crop y-coordinate from the form data, default to 0 if not provided
    crop_y = request.form.get('crop_y', type=int, default=0)
    # Capture the crop width from the form data, default to 100 if not provided
    crop_width = request.form.get('crop_width', type=int, default=100)
    # Capture the crop height from the form data, default to 100 if not provided
    crop_height = request.form.get('crop_height', type=int, default=100)
    # Capture the flip code from the form data, default to 1 if not provided
    flip_code = request.form.get('flip_code', type=int, default=1)
    # Capture the filter type from the form data
    filter_type = request.form.get('filter_type')
    # Capture the kernel size from the form data, default to 7 if not provided
    kernel_size = request.form.get('kernel_size', type=int, default=7)
    # Capture the edge detection algorithm from the form data
    edge_algorithm = request.form.get('edge_algorithm')
    # Capture the first threshold for edge detection from the form data, default to 100 if not provided
    threshold1 = request.form.get('threshold1', type=int, default=100)
    # Capture the second threshold for edge detection from the form data, default to 200 if not provided
    threshold2 = request.form.get('threshold2', type=int, default=200)
    # Capture the equalization type from the form data
    equalization_type = request.form.get('equalization_type')
    # Capture the clip limit for CLAHE from the form data, default to 2.0 if not provided
    clip_limit = request.form.get('clip_limit', type=float, default=2.0)
    # Capture the tile grid size for CLAHE from the form data, default to 8 if not provided
    tile_grid_size = request.form.get('tile_grid_size', type=int, default=8)
    # Capture the enhancement type from the form data
    enhancement_type = request.form.get('enhancement_type')
    # Capture the enhancement value from the form data, default to 1.0 if not provided
    enhancement_value = request.form.get('enhancement_value', type=float, default=1.0)
    # Capture the selected model from the form data
    model_name = request.form['model']

    # Check if the file has a valid filename
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        # Get the filename of the uploaded file
        filename = file.filename
        # Define the path to save the uploaded image
        image_path = os.path.join("uploads", filename)
        # If a file with the same name already exists, generate a unique filename
        if os.path.exists(image_path):
            filename = f"{uuid.uuid4().hex}_{filename}"
            image_path = os.path.join("uploads", filename)
        # Save the uploaded file to the specified path
        file.save(image_path)

        # Redirect to the results page with the provided parameters
        return redirect(url_for('results', filename=filename, result_option=result_option, color_space=color_space,
                                rotation_angle=rotation_angle, scale_factor=scale_factor,
                                crop_x=crop_x, crop_y=crop_y, crop_width=crop_width, crop_height=crop_height,
                                flip_code=flip_code, filter_type=filter_type, kernel_size=kernel_size,
                                edge_algorithm=edge_algorithm, threshold1=threshold1, threshold2=threshold2,
                                equalization_type=equalization_type, clip_limit=clip_limit,
                                tile_grid_size=tile_grid_size, model_name=model_name,
                                enhancement_type=enhancement_type, enhancement_value=enhancement_value))

# Define the results route to handle displaying processed images and results
@app.route('/results')
def results():
    try:
        # Check if any query parameters are provided
        if not request.args:
            # List all image files in the uploads directory
            image_files = [f for f in os.listdir('uploads') if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            return render_template('results.html', image_files=image_files)

        # Capture various parameters from the request arguments
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

        # Process the image using the selected model
        processed_image_path, detected_classes, processed_image_person_path = process_image(image_path, object_detection_model)
        segmented_image_path, segmentation_metrics = segment_image(image_path)

        # Apply various image processing techniques
        converted_image_paths = convert_color_space(image_path, color_space, result_option)
        transformed_image_path = transform_image(image_path, rotation_angle, scale_factor, crop_x, crop_y, crop_width, crop_height, flip_code)
        filtered_image_paths = apply_filter(image_path, filter_type, kernel_size, result_option)
        edge_detected_image_paths = apply_edge_detection(image_path, edge_algorithm, threshold1, threshold2, result_option)
        equalized_image_paths = apply_histogram_equalization(image_path, equalization_type, clip_limit, tile_grid_size, result_option)
        enhanced_image_paths = apply_image_enhancement(image_path, enhancement_type, enhancement_value, result_option)

        # Ensure converted_image_paths has enough elements
        converted_images = {}
        color_spaces = ['HSV', 'LAB', 'GRAY', 'RGB']
        if len(converted_image_paths) == 1:
            converted_images[color_space] = converted_image_paths[0]
        else:
            for i, cs in enumerate(color_spaces):
                if i < len(converted_image_paths):
                    converted_images[cs] = converted_image_paths[i]

        # Dictionary of color spaces with descriptions
        color_space_descriptions = {
            'HSV': 'Spațiul de culoare Hue, Saturation, and Value (HSV) reprezintă culorile în termeni de nuanță (hue), intensitate (saturation) și luminozitate (value). Este adesea folosit în procesarea imaginilor pentru segmentarea și filtrarea bazată pe culoare.',
            'LAB': 'Spațiul de culoare CIE L*a*b* (LAB) este un spațiu de culoare-oponent cu dimensiunea L pentru luminozitate și a și b pentru dimensiunile culoare-oponent. Este conceput pentru a aproxima vederea umană și este folosit pentru comparația și conversia culorilor.',
            'GRAY': 'Spațiul de culoare Grayscale reprezintă imaginile în nuanțe de gri, fiecare pixel corespunzând unei valori de intensitate. Este utilizat frecvent în sarcinile de procesare a imaginilor unde informațiile de culoare nu sunt necesare, cum ar fi detectarea marginilor și binarizarea.',
            'RGB': 'Spațiul de culoare RGB reprezintă imaginile folosind cele trei culori primare: Roșu, Verde și Albastru. Fiecare culoare este reprezentată printr-o combinație a acestor trei culori și este utilizat pe scară largă în tehnologiile de imagistică digitală și afișare.'
        }

        # Ensure filtered_image_paths has enough elements
        filtered_images = {}
        filter_types = ['gaussian', 'median']
        if len(filtered_image_paths) == 1:
            filtered_images[filter_type] = filtered_image_paths[0]
        else:
            for i, ft in enumerate(filter_types):
                if i < len(filtered_image_paths):
                    filtered_images[ft] = filtered_image_paths[i]

        # Dictionary of filter types with descriptions
        filter_type_descriptions = {
            'gaussian': 'Filtrul Gaussian Blur aplică o funcție Gaussiană imaginii, netezind-o prin reducerea zgomotului și a detaliilor. Este util pentru reducerea zgomotului și a detaliilor imaginii, fiind adesea folosit în pașii de preprocesare pentru a îmbunătăți performanța altor algoritmi.',
            'median': 'Filtrul Median Blur înlocuiește valoarea fiecărui pixel cu valoarea mediană a pixelilor vecini. Acest filtru este eficient în eliminarea zgomotului de tip sare și piper, păstrând în același timp marginile, fiind potrivit pentru imagini cu niveluri ridicate de zgomot.'
        }

        # Ensure edge_detected_image_paths has enough elements
        edge_detected_images = {}
        algorithms = ['canny', 'sobel', 'scharr', 'roberts', 'log']
        if len(edge_detected_image_paths) == 1:
            edge_detected_images[edge_algorithm] = edge_detected_image_paths[0]
        else:
            for i, algorithm in enumerate(algorithms):
                if i < len(edge_detected_image_paths):
                    edge_detected_images[algorithm] = edge_detected_image_paths[i]

        # Dictionary of edge detection algorithms with descriptions
        edge_algorithm_descriptions = {
            'canny': 'Algoritmul de detectare a marginilor Canny folosește un proces în mai multe etape pentru a detecta o gamă largă de margini în imagini. Implică reducerea zgomotului, calculul gradientului, suprimarea non-maximelor și urmărirea marginilor prin histerezis.',
            'sobel': 'Algoritmul de detectare a marginilor Sobel calculează gradientul intensității imaginii la fiecare pixel, subliniind regiunile cu frecvență spațială ridicată care corespund marginilor. Folosește convoluția cu kerneluri Sobel pentru a aproxima derivatele.',
            'scharr': 'Algoritmul de detectare a marginilor Scharr este o variație a operatorului Sobel care oferă o mai bună aproximare a magnitudinii gradientului, în special pentru marginile diagonale. Folosește kerneluri de convoluție specifice pentru a realiza acest lucru.',
            'roberts': 'Algoritmul de detectare a marginilor Roberts efectuează un calcul simplu și rapid al magnitudinii gradientului folosind o pereche de kerneluri de convoluție 2x2. Este deosebit de sensibil la zgomotul de înaltă frecvență și este cel mai potrivit pentru detectarea marginilor în imagini cu tranziții abrupte.',
            'log': 'Algoritmul de detectare a marginilor Laplacian of Gaussian (LoG) combină netezirea Gaussiană cu operatorul Laplacian pentru a detecta marginile. Mai întâi netezește imaginea pentru a reduce zgomotul și apoi aplică operatorul Laplacian pentru a evidenția regiunile cu schimbări rapide de intensitate.'
        }

        # Ensure equalized_image_paths has enough elements
        equalized_images = {}
        equalization_types = ['ahe', 'clahe']
        if len(equalized_image_paths) == 1:
            equalized_images[equalization_type] = equalized_image_paths[0]
        else:
            for i, et in enumerate(equalization_types):
                if i < len(equalized_image_paths):
                    equalized_images[et] = equalized_image_paths[i]

        # Dictionary of equalization types with descriptions
        equalization_type_descriptions = {
            'ahe': 'Egalizarea Adaptive Histogram (AHE) îmbunătățește contrastul unei imagini prin transformarea valorilor din histograma de intensitate. Funcționează prin împărțirea imaginii în regiuni mici și aplicarea egalizării histogramelor fiecărei regiuni în mod independent, ceea ce îmbunătățește contrastul local și scoate în evidență mai multe detalii în imagine.',
            'clahe': 'Egalizarea Adaptive Histogram Limitată de Contrast (CLAHE) este o variantă a AHE care previne amplificarea excesivă a zgomotului prin limitarea îmbunătățirii contrastului. Împarte imaginea în regiuni mici și aplică egalizarea histogramelor fiecărei regiuni, dar cu o limită de tăiere pentru a controla contrastul maxim. Acest lucru duce la o îmbunătățire mai echilibrată, reducând zgomotul și îmbunătățind vizibilitatea detaliilor.'
        }

        # Check if a person is detected and perform emotion detection
        emotions = []
        if 'person' in detected_classes:
            emotions = detect_emotions(image_path)

        # Ensure enhanced_image_paths has enough elements
        enhanced_images = {}
        enhancement_types = ['sharpen', 'denoise', 'brightness', 'contrast']
        if len(enhanced_image_paths) == 1:
            enhanced_images[enhancement_type] = enhanced_image_paths[0]
        else:
            for i, et in enumerate(enhancement_types):
                if i < len(enhanced_image_paths):
                    enhanced_images[et] = enhanced_image_paths[i]

        # Dictionary of enhancement types with descriptions
        enhancement_type_descriptions = {
            'sharpen': 'Ascuțește imaginea prin îmbunătățirea marginilor și a detaliilor fine, făcând imaginea să pară mai clară și mai definită.',
            'denoise': 'Reduce zgomotul din imagine, care poate fi cauzat de condiții de lumină scăzută sau setări ISO ridicate, rezultând o imagine mai netedă și mai curată.',
            'brightness': 'Ajustează luminozitatea imaginii, făcând-o mai luminoasă sau mai întunecată. Acest lucru poate ajuta la îmbunătățirea vizibilității în zonele subexpuse sau supraexpuse.',
            'contrast': 'Ajustează contrastul imaginii, îmbunătățind diferența dintre zonele luminoase și cele întunecate. Acest lucru poate face ca imaginea să pară mai vie și mai dinamică.'
        }

        # Render the results template with the processed images and descriptions
        return render_template('results.html', filename=filename, result_option=result_option,
                               model_name=model_name, processed_image=processed_image_path,
                               processed_image_person=processed_image_person_path, emotions=emotions,
                               segmented_image=segmented_image_path,
                               color_space=color_space, converted_images=converted_images,
                               color_space_descriptions=color_space_descriptions,
                               transformed_image=transformed_image_path,
                               filter_type=filter_type, filtered_images=filtered_images,
                               filter_type_descriptions=filter_type_descriptions,
                               equalization_type=equalization_type, equalized_images=equalized_images,
                               equalization_type_descriptions=equalization_type_descriptions,
                               enhancement_type=enhancement_type, enhanced_images=enhanced_images,
                               enhancement_type_descriptions=enhancement_type_descriptions,
                               detected_classes=detected_classes,
                               segmentation_metrics=segmentation_metrics,
                               edge_algorithm=edge_algorithm, edge_detected_images=edge_detected_images,
                               edge_algorithm_descriptions=edge_algorithm_descriptions)

    except ValueError as e:
        # Handle any ValueError exceptions by flashing the error message and redirecting to the index page
        flash(str(e))
        return redirect(url_for('index'))

def process_image(image_path, object_detection_model):
    # Open the image from the given path
    image = Image.open(image_path)
    # Perform object detection on the image
    results = object_detection_model(image_path)
    # Convert the image to a NumPy array
    image_np = np.array(image)

    # List to store detected class names
    detected_classes = []

    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Iterate over the detected objects
    for result in results.xyxy[0]:
        # Extract bounding box coordinates
        bbox = result[:4].tolist()
        # Extract class ID
        class_id = int(result[-1].item())
        # Extract confidence score
        confidence = result[-2].item()
        # Define rectangle color and thickness
        color = (0, 255, 0)
        thickness = 3
        # Define font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        # Draw the bounding box on the image
        cv2.rectangle(image_np, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, thickness)
        # Create label text with class name and confidence score
        label_text = f"{COCO_CLASSES[class_id]}: {confidence:.2f}"
        # Put the label text on the image
        cv2.putText(image_np, label_text, (int(bbox[0]), int(bbox[1]) - 10), font, font_scale, color, font_thickness)
        # Append the detected class name to the list
        detected_classes.append(COCO_CLASSES[class_id])

    # Generate the filename for the processed image
    processed_image_filename = f"processed_{os.path.basename(image_path)}"
    # Define the path to save the processed image
    processed_image_path = os.path.join("uploads", processed_image_filename)
    # Save the processed image
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

    # Generate the filename for the processed image with detected faces
    processed_image_person_filename = f"processed_person_{os.path.basename(image_path)}"
    # Define the path to save the processed image with detected faces
    processed_image_person_path = os.path.join("uploads", processed_image_person_filename)
    # Save the processed image with detected faces
    Image.fromarray(image_np).save(processed_image_person_path)

    # Return the filenames of the processed images and the list of detected classes
    return processed_image_filename, list(set(detected_classes)), processed_image_person_filename

def segment_image(image_path):
    # Open the image from the given path
    image = Image.open(image_path)
    # Preprocess the image and add a batch dimension
    image_tensor = preprocess(image).unsqueeze(0)
    # Perform segmentation using the pre-trained model without computing gradients
    with torch.no_grad():
        output = segmentation_model(image_tensor)['out'][0]
    # Get the predicted class for each pixel
    output_predictions = output.argmax(0).byte().cpu().numpy()
    # Convert the predictions to an image
    segmented_image = Image.fromarray(output_predictions)
    # Resize the segmented image to match the original image size
    segmented_image = segmented_image.resize(image.size)
    # Convert the segmented image to a NumPy array
    segmented_image_np = np.array(segmented_image)
    # Define a color map for the segmented classes
    color_map = {0: (0, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
    # Set the transparency level for blending
    transparency = 0.5
    # Define font properties for labeling
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    # Create an overlay for the segmented image
    overlay = np.array(image).copy()
    # Apply the color map to the segmented image
    for class_id, color in color_map.items():
        mask = (segmented_image_np == class_id)
        overlay[mask] = color
    # Blend the original image with the overlay
    blended_image = cv2.addWeighted(np.array(image), 1 - transparency, overlay, transparency, 0)
    # Draw contours around the segmented regions
    for class_id, color in color_map.items():
        mask = (segmented_image_np == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended_image, contours, -1, color, 2)
    # Label the segmented regions
    for class_id, color in color_map.items():
        mask = (segmented_image_np == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            label_text = f"Class {class_id}"
            cv2.putText(blended_image, label_text, (x, y - 10), font, font_scale, color, font_thickness)

    # Generate the filename for the segmented image
    segmented_image_filename = f"segmented_{os.path.basename(image_path)}"
    # Define the path to save the segmented image
    segmented_image_path = os.path.join("uploads", segmented_image_filename)
    # Save the blended image
    Image.fromarray(blended_image).save(segmented_image_path)

    # Reload the segmented image and calculate segmentation metrics
    segmented_image_np = np.array(Image.open(segmented_image_path))
    segmentation_metrics = calculate_segmentation_metrics(segmented_image_np)

    # Return the filename of the segmented image and the segmentation metrics
    return segmented_image_filename, segmentation_metrics


def transform_image(image_path, rotation_angle, scale_factor, crop_x, crop_y, crop_width, crop_height, flip_code):
    # Read the image from the given path
    image = cv2.imread(image_path)

    # Rotate the image if the rotation angle is not zero
    if rotation_angle != 0:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))

    # Scale the image if the scale factor is not 1.0
    if scale_factor != 1.0:
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # Crop the image if crop width and height are greater than zero
    if crop_width > 0 and crop_height > 0:
        image = image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

    # Flip the image if the flip code is valid (0, 1, or -1)
    if flip_code in [0, 1, -1]:
        image = cv2.flip(image, flip_code)

    # Generate the filename for the transformed image
    transformed_image_filename = f"transformed_{os.path.basename(image_path)}"
    # Define the path to save the transformed image
    transformed_image_path = os.path.join("uploads", transformed_image_filename)
    # Save the transformed image
    cv2.imwrite(transformed_image_path, image)
    print(f"Transformed image saved at: {transformed_image_path}")

    # Return the filename of the transformed image
    return transformed_image_filename

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
    # Get unique segment IDs and their counts in the segmented image
    unique, counts = np.unique(segmented_image_np, return_counts=True)
    # Calculate the total number of pixels in the image
    total_pixels = segmented_image_np.size
    # Initialize the metrics dictionary
    metrics = {
        "num_segments": len(unique),  # Number of unique segments
        "segments": []  # List to store metrics for each segment
    }
    # Iterate over each unique segment and its count
    for segment, count in zip(unique, counts):
        area = count  # Area of the segment (number of pixels)
        percentage = (count / total_pixels) * 100  # Percentage of the image covered by the segment
        # Append the segment metrics to the list
        metrics["segments"].append({
            "segment_id": int(segment),  # Segment ID
            "area": area,  # Area of the segment
            "percentage": percentage  # Percentage of the image covered by the segment
        })
    # Return the calculated metrics
    return metrics

def detect_emotions(image_path):
    """
    Detect emotions on faces in the given image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        list: A list of dictionaries containing detected emotions and their probabilities.
    """
    # Analyze the image to detect emotions using DeepFace
    results_emotions = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)

    # Return the detected emotions and their probabilities
    return results_emotions

def convert_color_space(image_path, color_space, result_option):
    # Read the image from the given path
    image = cv2.imread(image_path)
    # Initialize an empty list to store the results
    results = []

    if result_option == 'all':
        # Return all color space conversions
        color_spaces = ['HSV', 'LAB', 'GRAY', 'RGB']
        for cs in color_spaces:
            if cs == 'HSV':
                # Convert the image to HSV color space
                converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif cs == 'LAB':
                # Convert the image to LAB color space
                converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            elif cs == 'GRAY':
                # Convert the image to Grayscale
                converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                # Keep the image in RGB color space
                converted_image = image

            # Generate the filename for the converted image
            converted_image_filename = f"converted_{cs}_{os.path.basename(image_path)}"
            # Define the path to save the converted image
            converted_image_path = os.path.join("uploads", converted_image_filename)
            # Save the converted image
            cv2.imwrite(converted_image_path, converted_image)
            # Append the filename to the results list
            results.append(converted_image_filename)
    else:
        # Return single color space conversion
        if color_space == 'HSV':
            # Convert the image to HSV color space
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == 'LAB':
            # Convert the image to LAB color space
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif color_space == 'GRAY':
            # Convert the image to Grayscale
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # Keep the image in RGB color space
            converted_image = image

        # Generate the filename for the converted image
        converted_image_filename = f"converted_{os.path.basename(image_path)}"
        # Define the path to save the converted image
        converted_image_path = os.path.join("uploads", converted_image_filename)
        # Save the converted image
        cv2.imwrite(converted_image_path, converted_image)
        # Append the filename to the results list
        results.append(converted_image_filename)

    # Return the list of converted image filenames
    return results

def apply_filter(image_path, filter_type, kernel_size, result_option):
    # Read the image from the given path
    image = cv2.imread(image_path)
    # Initialize an empty list to store the results
    results = []

    if result_option == 'all':
        # Return all filter types
        filter_types = ['gaussian', 'median']
        for ft in filter_types:
            if ft == 'gaussian':
                # Apply Gaussian Blur filter to the image
                filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            elif ft == 'median':
                # Apply Median Blur filter to the image
                filtered_image = cv2.medianBlur(image, kernel_size)
            # Generate the filename for the filtered image
            filtered_image_filename = f"filtered_{ft}_{os.path.basename(image_path)}"
            # Define the path to save the filtered image
            filtered_image_path = os.path.join("uploads", filtered_image_filename)
            # Save the filtered image
            cv2.imwrite(filtered_image_path, filtered_image)
            # Append the filename to the results list
            results.append(filtered_image_filename)
    else:
        # Return single filter type
        if filter_type == 'gaussian':
            # Apply Gaussian Blur filter to the image
            filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif filter_type == 'median':
            # Apply Median Blur filter to the image
            filtered_image = cv2.medianBlur(image, kernel_size)
        else:
            # If no valid filter type is provided, keep the original image
            filtered_image = image
        # Generate the filename for the filtered image
        filtered_image_filename = f"filtered_{os.path.basename(image_path)}"
        # Define the path to save the filtered image
        filtered_image_path = os.path.join("uploads", filtered_image_filename)
        # Save the filtered image
        cv2.imwrite(filtered_image_path, filtered_image)
        # Append the filename to the results list
        results.append(filtered_image_filename)

    # Return the list of filtered image filenames
    return results

def apply_edge_detection(image_path, edge_algorithm, threshold1, threshold2, result_option):
    # Read the image from the given path in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Initialize an empty list to store the results
    results = []

    if result_option == 'all':
        # Return all edge detection algorithms
        edge_algorithms = ['canny', 'sobel', 'scharr', 'roberts', 'log']
        for ea in edge_algorithms:
            if ea == 'canny':
                # Apply Canny edge detection
                edges = cv2.Canny(image, threshold1, threshold2)
            elif ea == 'sobel':
                # Apply Sobel edge detection
                edges = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
            elif ea == 'scharr':
                # Apply Scharr edge detection
                edges = cv2.Scharr(image, cv2.CV_64F, 1, 0)
            elif ea == 'roberts':
                # Apply Roberts edge detection
                kernel = np.array([[1, 0], [0, -1]], dtype=np.float32)
                edges = cv2.filter2D(image, -1, kernel)
            elif ea == 'log':
                # Apply Laplacian of Gaussian (LoG) edge detection
                edges = cv2.Laplacian(image, cv2.CV_64F)
            # Generate the filename for the edge-detected image
            edge_detected_image_filename = f"edge_detected_{ea}_{os.path.basename(image_path)}"
            # Define the path to save the edge-detected image
            edge_detected_image_path = os.path.join("uploads", edge_detected_image_filename)
            # Save the edge-detected image
            cv2.imwrite(edge_detected_image_path, edges)
            # Append the filename to the results list
            results.append(edge_detected_image_filename)
    else:
        # Return single edge detection algorithm
        if edge_algorithm == 'canny':
            # Apply Canny edge detection
            edges = cv2.Canny(image, threshold1, threshold2)
        elif edge_algorithm == 'sobel':
            # Apply Sobel edge detection
            edges = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        elif edge_algorithm == 'scharr':
            # Apply Scharr edge detection
            edges = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        elif edge_algorithm == 'roberts':
            # Apply Roberts edge detection
            kernel = np.array([[1, 0], [0, -1]], dtype=np.float32)
            edges = cv2.filter2D(image, -1, kernel)
        elif edge_algorithm == 'log':
            # Apply Laplacian of Gaussian (LoG) edge detection
            edges = cv2.Laplacian(image, cv2.CV_64F)
        else:
            # If no valid edge detection algorithm is provided, keep the original image
            edges = image
        # Generate the filename for the edge-detected image
        edge_detected_image_filename = f"edge_detected_{os.path.basename(image_path)}"
        # Define the path to save the edge-detected image
        edge_detected_image_path = os.path.join("uploads", edge_detected_image_filename)
        # Save the edge-detected image
        cv2.imwrite(edge_detected_image_path, edges)
        # Append the filename to the results list
        results.append(edge_detected_image_filename)

    # Return the list of edge-detected image filenames
    return results

def apply_histogram_equalization(image_path, equalization_type, clip_limit, tile_grid_size, result_option):
    # Read the image from the given path in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Initialize an empty list to store the results
    results = []

    if result_option == 'all':
        # Return all equalization types
        equalization_types = ['ahe', 'clahe']
        for et in equalization_types:
            if et == 'ahe':
                # Apply Adaptive Histogram Equalization (AHE)
                equalized_image = cv2.equalizeHist(image)
            elif et == 'clahe':
                # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
                equalized_image = clahe.apply(image)
            # Generate the filename for the equalized image
            equalized_image_filename = f"equalized_{et}_{os.path.basename(image_path)}"
            # Define the path to save the equalized image
            equalized_image_path = os.path.join("uploads", equalized_image_filename)
            # Save the equalized image
            cv2.imwrite(equalized_image_path, equalized_image)
            # Append the filename to the results list
            results.append(equalized_image_filename)
    else:
        # Return single equalization type
        if equalization_type == 'ahe':
            # Apply Adaptive Histogram Equalization (AHE)
            equalized_image = cv2.equalizeHist(image)
        elif equalization_type == 'clahe':
            # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            equalized_image = clahe.apply(image)
        else:
            # If no valid equalization type is provided, keep the original image
            equalized_image = image
        # Generate the filename for the equalized image
        equalized_image_filename = f"equalized_{os.path.basename(image_path)}"
        # Define the path to save the equalized image
        equalized_image_path = os.path.join("uploads", equalized_image_filename)
        # Save the equalized image
        cv2.imwrite(equalized_image_path, equalized_image)
        # Append the filename to the results list
        results.append(equalized_image_filename)

    # Return the list of equalized image filenames
    return results

def apply_image_enhancement(image_path, enhancement_type, enhancement_value, result_option):
    # Open the image from the given path
    image = Image.open(image_path)
    # Initialize an empty list to store the results
    results = []

    if result_option == 'all':
        # Return all enhancement types
        enhancement_types = ['sharpen', 'denoise', 'brightness', 'contrast']
        for et in enhancement_types:
            if et == 'sharpen':
                # Apply sharpening enhancement
                enhancer = ImageEnhance.Sharpness(image)
            elif et == 'denoise':
                # Apply denoising enhancement (placeholder, use actual denoising method)
                enhancer = ImageEnhance.Sharpness(image)
            elif et == 'brightness':
                # Apply brightness enhancement
                enhancer = ImageEnhance.Brightness(image)
            elif et == 'contrast':
                # Apply contrast enhancement
                enhancer = ImageEnhance.Contrast(image)
            # Enhance the image with the specified enhancement value
            enhanced_image = enhancer.enhance(enhancement_value)
            # Generate the filename for the enhanced image
            enhanced_image_filename = f"enhanced_{et}_{os.path.basename(image_path)}"
            # Define the path to save the enhanced image
            enhanced_image_path = os.path.join("uploads", enhanced_image_filename)
            # Save the enhanced image
            enhanced_image.save(enhanced_image_path)
            # Append the filename to the results list
            results.append(enhanced_image_filename)
    else:
        # Return single enhancement type
        if enhancement_type == 'sharpen':
            # Apply sharpening enhancement
            enhancer = ImageEnhance.Sharpness(image)
        elif enhancement_type == 'denoise':
            # Apply denoising enhancement (placeholder, use actual denoising method)
            enhancer = ImageEnhance.Sharpness(image)
        elif enhancement_type == 'brightness':
            # Apply brightness enhancement
            enhancer = ImageEnhance.Brightness(image)
        elif enhancement_type == 'contrast':
            # Apply contrast enhancement
            enhancer = ImageEnhance.Contrast(image)
        else:
            # No valid enhancement type provided
            enhancer = None

        if enhancer:
            # Enhance the image with the specified enhancement value
            enhanced_image = enhancer.enhance(enhancement_value)
        else:
            # Keep the original image if no valid enhancement type is provided
            enhanced_image = image

        # Generate the filename for the enhanced image
        enhanced_image_filename = f"enhanced_{os.path.basename(image_path)}"
        # Define the path to save the enhanced image
        enhanced_image_path = os.path.join("uploads", enhanced_image_filename)
        # Save the enhanced image
        enhanced_image.save(enhanced_image_path)
        # Append the filename to the results list
        results.append(enhanced_image_filename)

    # Return the list of enhanced image filenames
    return results

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Serve the file from the 'uploads' directory with the given filename
    return send_from_directory('uploads', filename)

if __name__ == "__main__":
    # Check if the 'uploads' directory exists
    if not os.path.exists('uploads'):
        # Create the 'uploads' directory if it does not exist
        os.makedirs('uploads')
    # Run the Flask application in debug mode
    app.run(debug=True)