from flask import Flask, render_template, jsonify, request
from werkzeug.utils import secure_filename
from classifier import *
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Secure the filename
    filename = secure_filename(file.filename)
    
    # Save the file to the specified folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    classifier = Classifier()
    category_name, score = classifier.forward(file_path)
    
    return jsonify({"message": f"File successfully uploaded. {category_name} {score}", "filename": filename}), 200
