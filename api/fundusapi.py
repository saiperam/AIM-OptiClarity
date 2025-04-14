import torch
import torch.nn as nn
import torchvision.models as models
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import torchvision.transforms.v2 as transforms
import numpy as np
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

# Initialize FastAPI app
app = FastAPI()

# Allow CORS from your frontend (localhost:5173)
origins = [
    "http://localhost:5173",  # Your frontend URL
    "http://localhost:3000",  # If you are using Create React App on this port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows all origins from the specified list
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define the class to store the prediction response
class PredictionResponse(BaseModel):
    predictions: list
    prediction: str


# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_url = "https://huggingface.co/Abrar21/FundusModel/resolve/main/best_fundus_model2.pth"

class FundusModel(nn.Module):
    def __init__(self, num_classes=8):
        super(FundusModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Load the trained model weights
model = FundusModel().to(device)

checkpoint_path = hf_hub_download(
    repo_id="Abrar21/FundusModel",
    filename="best_fundus_model2.pth"
)

state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Data Preprocessing Pipeline
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_image(image):
    image = data_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float()
    return preds.cpu().numpy(), probs.cpu().numpy()



@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Extract just the filename (without path)
        original_filename = file.filename

        # Read and preprocess the image
        img = Image.open(BytesIO(await file.read())).convert('RGB')
        preds, probs = predict_image(img)

        labels = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'Age related Macular Degeneration',
                  'Hypertension', 'Pathological Myopia', 'Other diseases/abnormalities']

        prediction = {
            'predictions': [{'label': labels[i], 'confidence': float(probs[0][i])} for i in range(len(preds[0]))],
            'prediction': labels[np.argmax(probs)],
        }

        return prediction

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)
