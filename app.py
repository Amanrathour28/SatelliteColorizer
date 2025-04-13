import os
import uuid
from flask import Flask, render_template, request, redirect, send_from_directory
from PIL import Image
import numpy as np
import onnxruntime as ort
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
SEGMENT_FOLDER = 'static/segments'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, SEGMENT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Load ONNX models
color_model_path = r'C:\Users\amanr\Desktop\satellite-colorizer\best_model.onnx'
seg_model_path = r'C:\Users\amanr\Desktop\satellite-colorizer\segmentation_model.onnx'

color_session = ort.InferenceSession(color_model_path, providers=['CPUExecutionProvider'])
seg_session = ort.InferenceSession(seg_model_path, providers=['CPUExecutionProvider'])

# Allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Colorize image
def colorize_image(image_path, output_path):
    img = Image.open(image_path).convert('L').resize((256, 256))
    img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, 256, 256, 1)

    input_name = color_session.get_inputs()[0].name
    output_name = color_session.get_outputs()[0].name
    prediction = color_session.run([output_name], {input_name: img_array})[0]

    output_image = ((prediction[0] + 1) * 127.5).astype(np.uint8)
    Image.fromarray(output_image).save(output_path)

# Segment image
def segment_image(colorized_path, segmented_path):
    img = cv2.imread(colorized_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img_input = img.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)  # (1, 256, 256, 3)

    input_name = seg_session.get_inputs()[0].name
    output = seg_session.run(None, {input_name: img_input})[0]  # (1, 256, 256, 7)
    pred = np.argmax(output.squeeze(), axis=-1).astype(np.uint8)

    # Fixed color map & label names (matches notebook)
    color_map = {
        0: (255, 0, 0),     # Urban Land
        1: (0, 255, 0),     # Agriculture Land
        2: (160, 82, 45),   # Rangeland
        3: (0, 100, 0),     # Forest Land
        4: (0, 255, 255),   # Water
        5: (255, 255, 0),   # Barren Land
        6: (128, 128, 128)  # Unknown
    }

    label_names = {
        0: "Urban Land",
        1: "Agriculture Land",
        2: "Rangeland",
        3: "Forest Land",
        4: "Water",
        5: "Barren Land",
        6: "Unknown"
    }

    segmented_img = np.zeros((256, 256, 3), dtype=np.uint8)
    for label, color in color_map.items():
        segmented_img[pred == label] = color

    Image.fromarray(segmented_img).save(segmented_path)

    total = pred.size
    breakdown = {
        label_names[label]: np.sum(pred == label) / total * 100
        for label in range(7)
    }

    return [f"{label}: {percent:.2f}%" for label, percent in breakdown.items()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    if file and allowed_file(file.filename):
        filename = f"{uuid.uuid4().hex}.png"
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        colorized_filename = f"colorized_{filename}"
        colorized_path = os.path.join(OUTPUT_FOLDER, colorized_filename)
        colorize_image(upload_path, colorized_path)

        return render_template('results.html',
                               original_image=filename,
                               colorized_image=colorized_filename,
                               colorized_filename=colorized_filename)
    return redirect('/')

@app.route('/segment')
def segment():
    colorized_files = sorted(
        os.listdir(OUTPUT_FOLDER),
        key=lambda x: os.path.getctime(os.path.join(OUTPUT_FOLDER, x)),
        reverse=True
    )
    if not colorized_files:
        return redirect('/')

    colorized_filename = colorized_files[0]
    colorized_path = os.path.join(OUTPUT_FOLDER, colorized_filename)

    segmented_filename = f"segmented_{colorized_filename}"
    segmented_path = os.path.join(SEGMENT_FOLDER, segmented_filename)

    percentages = segment_image(colorized_path, segmented_path)

    return render_template('segmentation_result.html',
                           colorized_filename=colorized_filename,
                           segmented_filename=segmented_filename,
                           percentages=percentages)

@app.route('/download/<folder>/<filename>')
def download_file(folder, filename):
    folder_map = {
        'outputs': OUTPUT_FOLDER,
        'segments': SEGMENT_FOLDER,
        'uploads': UPLOAD_FOLDER
    }
    if folder in folder_map:
        return send_from_directory(folder_map[folder], filename, as_attachment=True)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)


    
