from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2

# Creating a Flask Instance
app = Flask(__name__)

IMAGE_SIZE = (150, 150)
UPLOAD_FOLDER = 'static\\uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print("Loading Pre-trained Model ...")
try:
    model = load_model('model.h5')
    print("Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None


def image_preprocessor(path):
    try:
        print('Processing Image ...')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        
        currImg_BGR = cv2.imread(path)
        if currImg_BGR is None:
            raise ValueError(f"Could not load image from: {path}")
        
        currImg_RGB = cv2.cvtColor(currImg_BGR, cv2.COLOR_BGR2RGB)
        currImg = cv2.resize(currImg_RGB, IMAGE_SIZE)
        currImg = currImg / 255.0
        currImg = np.reshape(currImg, (1, 150, 150, 3))
        return currImg
    except Exception as e:
        print(f'Error in image preprocessing: {str(e)}')
        raise


def model_pred(image):
    try:
        if model is None:
            raise ValueError("Model not loaded properly")
        
        predictions = model.predict(image)
        print(f"Raw prediction output: {predictions}")  # log it for debugging

        # For binary classification: returns probability between 0 and 1
        prediction = predictions[0][0]

        # Log probability and result
        print(f"Probability: {prediction:.4f}")

        # Threshold-based decision
        result = 1 if prediction > 0.5 else 0
        print("Predicted:", "PNEUMONIA" if result == 1 else "NORMAL")
        return result

    except Exception as e:
        print(f'Error in model prediction: {str(e)}')
        raise



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files.get('imageFile')
        if not file or file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                imgPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(imgPath)
                
                image = image_preprocessor(imgPath)
                pred = model_pred(image)
                return render_template('index.html', name=filename, result=pred)
            except Exception as e:
                flash(f'Error: {str(e)}')
                return redirect(request.url)
    return render_template('index.html', name=None, result=None)


if __name__ == '__main__':
    app.run(debug=True)
