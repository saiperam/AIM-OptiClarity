import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as transforms
from PIL import Image
import pandas as pd
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

        # Store file paths and labels
        self.image_paths = [os.path.join(self.img_dir, f"{row[18]}") for row in self.labels_df.values]
        self.labels = self.labels_df['target'].apply(lambda x: torch.tensor(eval(x), dtype=torch.float32))

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),  
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  
    transforms.RandomAffine(degrees=0, shear=0.2),  
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  
    transforms.RandomHorizontalFlip(),  
    transforms.ToImage(),  
    transforms.ToDtype(torch.float32, scale=True),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Dataset
train_dataset = FundusDataset(
    img_dir='input/ocular-disease-recognition-odir5k/ODIR-5K/Training Images',
    labels_file='input/ocular-disease-recognition-odir5k/full_df.csv',
    transform=data_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

# Define Model
class FundusModel(nn.Module):
    def __init__(self, num_classes=8):
        super(FundusModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = True  # Freeze/Unfreeze all layers
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize Model
model = FundusModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-7)

# Learning Rate Finder Function
def find_lr(model, train_loader, criterion, optimizer, init_lr=1e-7, final_lr=1, beta=0.98):
    num = len(train_loader) - 1
    mult = (final_lr / init_lr) ** (1/num)
    lr = init_lr
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = float('inf')
    losses = []
    log_lrs = []
    
    model.train()
    
    for batch_num, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_num + 1}/{len(train_loader)}")
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (batch_num + 1))
        
        if batch_num > 1 and smoothed_loss > best_loss * 4:
            break  # Stop if loss explodes
        
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        
        losses.append(smoothed_loss)
        log_lrs.append(lr)
        
        loss.backward()
        optimizer.step()
        
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

    # Convert to NumPy arrays
    losses = np.array(losses)
    log_lrs = np.array(log_lrs)

    # Find the learning rate with the steepest drop
    min_grad_index = np.gradient(losses).argmin()
    best_lr = log_lrs[min_grad_index]

    # Plot the loss vs learning rate
    plt.figure(figsize=(10, 5))
    plt.plot(log_lrs, losses, label="Loss")
    plt.axvline(best_lr, color='r', linestyle='--', label=f"Best LR: {best_lr:.2e}")
    plt.xscale('log')
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.legend()
    plt.show()

    print(f"Suggested Learning Rate: {best_lr:.2e}")
    return best_lr

# Run the LR Finder
if __name__ == "__main__":
    best_lr = find_lr(model, train_loader, criterion, optimizer)
    print(f"Best learning rate determined: {best_lr}")

