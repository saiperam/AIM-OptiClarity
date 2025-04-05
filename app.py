
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import tensorflow as tf

print(tf.__version__)  
print(tf.keras)  


app = Flask(__name__)

model = tf.keras.models.load_model('keratoconus_detection_model3-23.h5')  

@app.route('/')
def home():
    return "Flask app is running! Use POST /predict to send image data."
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((224, 224)) 
        image = np.array(image) / 255.0  
        
        image = np.expand_dims(image, axis=0)
        
        prediction = model.predict(image)
        
        class_label = np.argmax(prediction)
        if class_label == 0:
            class_label = "Keratoconus"
        elif class_label == 1:
            class_label = "Normal"
        else:
            class_label = "Suspect"
        accuracy = float(np.max(prediction) * 100)
        
        return jsonify({
            'predicted_class': class_label,
            'accuracyw': accuracy
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'app.py is up to date'})


if __name__ == '__main__':
    app.run(debug=True)
