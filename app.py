
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import tensorflow as tf

print(tf.__version__)  # Check TensorFlow version
print(tf.keras)  # Check if Keras is available under tensorflow


# Initialize the Flask application
app = Flask(__name__)

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
        
        # Convert prediction to a Python float to avoid serialization error
        accuracy = float(np.max(prediction) * 100)
        
        # Return the prediction and accuracy
        return jsonify({
            'predicted_class': class_label,
            'accuracy': accuracy
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
