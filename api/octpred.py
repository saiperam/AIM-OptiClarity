import torch
import uvicorn
import pathlib
import numpy as np
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from torchvision.models import resnet50, inception_v3, efficientnet_b3
from torch import nn
from PIL import Image

# Initialize FastAPI app
app = FastAPI()

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define class names
CLASS_NAMES = ["NORMAL", "MH", "DRUSEN", "DR", "DME", "CSR", "CNV", "AMD"]
NUM_CLASSES = len(CLASS_NAMES)

# Load models
inc_v3 = inception_v3(pretrained=False)
resnet = resnet50(pretrained=False)
eff_net = efficientnet_b3(pretrained=False)

# Modify classifier layers
inc_v3.fc = nn.Linear(inc_v3.fc.in_features, NUM_CLASSES)
resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)
eff_net.classifier[1] = nn.Linear(eff_net.classifier[1].in_features, NUM_CLASSES)


# Define ranker model
class Ranker(nn.Module):
    def __init__(self, num_classes):
        super(Ranker, self).__init__()
        self.fc_layer = nn.Linear(in_features=3 * num_classes, out_features=num_classes)

    def forward(self, eff_output, inc_output, resnet_output):
        return self.fc_layer(torch.cat([eff_output, inc_output, resnet_output], dim=1))


ranker = Ranker(NUM_CLASSES)

# Load saved weights (Correcting key names)
model_path = pathlib.Path("oct_model.pth")
state_dict = torch.load(model_path, map_location=device)

# Using correct key names from state_dict
inc_v3.load_state_dict(state_dict["inc_v3_state_dict"])
resnet.load_state_dict(state_dict["resnet_state_dict"])
eff_net.load_state_dict(state_dict["eff_net_state_dict"])
ranker.load_state_dict(state_dict["ranker_state_dict"])

# Move models to device
inc_v3.to(device).eval()
resnet.to(device).eval()
eff_net.to(device).eval()
ranker.to(device).eval()

# Extract the computed mean and std from the tensors
computed_mean = [0.2096, 0.2096, 0.2096]
computed_std = [0.1558, 0.1558, 0.1558]

# Define image preprocessing using the computed mean and std
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=computed_mean, std=computed_std),  # Using your dataset-specific mean and std
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        eff_out = eff_net(image)
        inc_out = inc_v3(image)
        resnet_out = resnet(image)
        final_output = ranker(eff_out, inc_out, resnet_out)
        prediction = torch.argmax(final_output, dim=1).item()
        confidence = torch.softmax(final_output, dim=1).max().item()

    return {"prediction": CLASS_NAMES[prediction], "confidence": confidence}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)
