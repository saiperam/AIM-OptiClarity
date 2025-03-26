from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('corneal_ulcer_model.h5') #loading trained model

def preprocess(image):
  image = cv2.resize(image, (224,224))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = image/255.0

  image = np.expand_dims(image, axis=0)    # batch normalizing --> (1,224,224,3)

  return image

@app.route('/predict', methods = ['POST'] )
def predict():
  if 'image' not in request.files:
    return jsonify({"error":"no image uploaded"}), 400
  
  file = request.files['image']    # key is image in post request, value is the file
  file_bytes = np.frombuffer(file.read(),np.uint8)    # getting file bytes
  img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)     # converting into image array

  img = preprocess(img) 

  prediction = model.predict(img)

  decoded_prediction = prediction.argmax(axis=1)   # highest prediciton

  prediction_list = decoded_prediction.tolist()    # converting to list so its json serializable

  return jsonify({"prediction": prediction_list})

@app.route("/get", methods = ['GET'])   # testing server
def hello_world():
  return "<p>Hello World!</p>"

if __name__=='__main__':
  app.run(host='0.0.0.0',port=6000,debug=True)