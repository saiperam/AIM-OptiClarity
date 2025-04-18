from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS
import numpy as np
import io
import tensorflow as tf
from huggingface_hub import hf_hub_download
import os

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Download the .h5 model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="AAZ21/keratoconus_corneal_topo",
    filename="keratoconus_detection_model3-23.h5"
)

# Load the model
model = tf.keras.models.load_model(model_path)

@app.route('/')
def home():
    return "Flask app is running! Use POST /predict to send image data."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['file']

        # Read the image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = image.resize((224, 224))  # Resize as needed
        image = np.array(image) / 255.0  # Normalize the image

        # Add batch dimension (for model input)
        image = np.expand_dims(image, axis=0)

        # Make prediction
        prediction = model.predict(image)

        # Process the prediction (e.g., get class label)
        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        # Map index to label
        labels = ["Keratoconus", "Normal", "Suspect"]
        class_label = labels[class_index]

        # Return the prediction and accuracy
        return jsonify({
            'prediction': class_label,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'app.py is up to date'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8003)


























"""from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS
import numpy as np
import io
import tensorflow as tf

print(tf.__version__)  # Check TensorFlow version
print(tf.keras)  # Check if Keras is available under tensorflow

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Load your model (ensure it's saved and accessible)
model = tf.keras.models.load_model('keratoconus_detection_model3-23.h5')  # Adjust model path


@app.route('/')
def home():
    return "Flask app is running! Use POST /predict to send image data."


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['file']

        # Read the image
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((224, 224))  # Resize as needed
        image = np.array(image) / 255.0  # Normalize the image

        # Add batch dimension (for model input)
        image = np.expand_dims(image, axis=0)

        # Make prediction
        prediction = model.predict(image)

        # Process the prediction (e.g., get class label)
        class_label = np.argmax(prediction)
        if class_label == 0:
            class_label = "Keratoconus"
        elif class_label == 1:
            class_label = "Normal"
        else:
            class_label = "Suspect"
        accuracy = float(np.max(prediction))

        # Return the prediction and accuracy
        return jsonify({
            'prediction': class_label,
            'confidence': accuracy
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'app.py is up to date'})

if __name__ == '__main__':
    app.run(host='localhost', debug=True, port=8003)"""
