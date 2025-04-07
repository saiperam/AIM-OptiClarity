import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torchvision.models as models
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Custom Dataset Class
class FundusDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        self.img_dir = img_dir
        self.labels_file = labels_file
        self.transform = transform
        if not os.path.exists(self.labels_file):
            raise FileNotFoundError(f"Labels file not found at {os.path.abspath(self.labels_file)}")
        self.labels_df = pd.read_csv(self.labels_file)

        # Store file paths
        self.image_paths_left = [os.path.join(self.img_dir, f"{row[4]}") for row in self.labels_df.values]

        # Parse the target column which contains the label lists (as string) and convert them to tensor
        self.labels = self.labels_df['target'].apply(lambda x: torch.tensor(eval(x), dtype=torch.float32))

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name_left = self.image_paths_left[idx]
        image_left = Image.open(img_name_left).convert('RGB')

        if self.transform:
            image_left = self.transform(image_left)
        
        return image_left, self.labels[idx]

# Define the Model
class FundusModel(nn.Module):
    def __init__(self, num_classes=8):
        super(FundusModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_metrics(outputs, labels, threshold=0.5):
    preds = (torch.sigmoid(outputs) > threshold).float()
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')

    return accuracy, precision, recall, f1

def test_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy, precision, recall, f1 = calculate_metrics(torch.tensor(all_preds), torch.tensor(all_labels))

    return avg_loss, accuracy, precision, recall, f1

if __name__ == '__main__':
    # Data Transformations
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the full dataset
    full_dataset = FundusDataset('input/ocular-disease-recognition-odir5k/ODIR-5K/Training Images', 
                                 'input/ocular-disease-recognition-odir5k/full_df.csv', 
                                 transform=data_transform)

    # Data Loader
    num_workers = os.cpu_count()
    test_loader = DataLoader(full_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # Load the trained model
    model = FundusModel().to(device)
    model.load_state_dict(torch.load('best_fundus_model.pth'))

    # Define Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Test the model
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = test_model(model, test_loader, criterion)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    