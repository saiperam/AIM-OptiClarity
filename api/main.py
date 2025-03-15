import numpy as np
from cffi.cffi_opcode import CLASS_NAME
from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.inception_v3 import preprocess_input

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../model/inceptionv3.keras")
CLASS_NAMES = ["AMD", "CNV", "CSR", "DME", "Diabetic Retinopathy", "DRUSEN", "MH", "NORMAL"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

"""def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image"""
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")  # Ensure it's RGB
    image = image.resize((299, 299))  # Resize to 299x299
    image = np.array(image, dtype=np.float32)  # Convert to NumPy array
    image = preprocess_input(image)  # Scale to [-1, 1]
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0).astype(np.float32)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8002)