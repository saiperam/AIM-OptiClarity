import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class FundusDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        self.img_dir = img_dir
        self.labels_file = labels_file
        self.transform = transform

        if not os.path.exists(self.labels_file):
            raise FileNotFoundError(f"Labels file not found at {os.path.abspath(self.labels_file)}")
        
        self.labels_df = pd.read_csv(self.labels_file)
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
    
class FundusModel(nn.Module):
    def __init__(self, num_classes=8):
        super(FundusModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False  # Freeze all layers initially
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_accuracies(outputs, labels, threshold=0.5):
    '''
    Calculates and returns the image and label accuracies
    '''
    preds = (torch.sigmoid(outputs) > threshold).float()
    image_correct = (preds == labels).float().mean()
    label_correct = (torch.sum(preds == labels, dim=1) == labels.size(1)).float().mean()
    return image_correct.item(), label_correct.item()

def run_epoch(model, data_loader, criterion, optimizer=None, epoch=None, mode='train'):
    '''
    Runs one epoch of training or validation
    '''
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    
    total_loss, total_image_accuracy, total_label_accuracy = 0.0, 0.0, 0.0
    progress_bar = tqdm(data_loader, desc=f"{mode.capitalize()} Epoch {epoch+1}", unit="batch")
    
    with torch.set_grad_enabled(is_training):
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            image_accuracy, label_accuracy = calculate_accuracies(outputs, labels)
            total_loss += loss.item()
            total_image_accuracy += image_accuracy
            total_label_accuracy += label_accuracy

            progress_bar.set_postfix({
                'Loss': total_loss / (progress_bar.n + 1),
                'Image Acc': total_image_accuracy / (progress_bar.n + 1),
                'Label Acc': total_label_accuracy / (progress_bar.n + 1)
            })

    avg_loss = total_loss / len(data_loader)
    avg_image_accuracy = total_image_accuracy / len(data_loader)
    avg_label_accuracy = total_label_accuracy / len(data_loader)
    
    return avg_loss, avg_image_accuracy, avg_label_accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15, fine_tune_epochs=5):
    '''
    Trains the model and fine-tunes it at the end of training
    '''
    print("Training the model...")
    best_val_accuracy = 0.0
    best_val_loss = 0.0
    history = {
        "train_loss": [],
        "train_image_acc": [],
        "train_label_acc": [],
        "val_loss": [],
        "val_image_acc": [],
        "val_label_acc": []
    }

    best_model_path = 'best_fundus_model2.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded the best model from previous training.")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")

        train_loss, train_image_accuracy, train_label_accuracy = run_epoch(model, train_loader, criterion, optimizer, epoch, mode='train')
        
        history["train_loss"].append(train_loss)
        history["train_image_acc"].append(train_image_accuracy)
        history["train_label_acc"].append(train_label_accuracy)

        val_loss, val_image_accuracy, val_label_accuracy = run_epoch(model, val_loader, criterion, optimizer=None, epoch=epoch, mode='validation')

        history["val_loss"].append(val_loss)
        history["val_image_acc"].append(val_image_accuracy)
        history["val_label_acc"].append(val_label_accuracy)

        print(f"Train Loss: {train_loss:.4f}, Train Image Accuracy: {train_image_accuracy:.4f}, Train Label Accuracy: {train_label_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Image Accuracy: {val_image_accuracy:.4f}, Validation Label Accuracy: {val_label_accuracy:.4f}")
        
        scheduler.step(val_loss) 
        print(f"New Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")

        if val_label_accuracy > best_val_accuracy:
            best_val_accuracy = val_label_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch+1}")

    print("Fine-tuning the entire model...")
    for param in model.model.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=4e-05, weight_decay=1e-2)  

    for epoch in range(fine_tune_epochs):
        print(f"Fine-tuning Epoch {epoch+1}/{fine_tune_epochs}")
        train_loss, train_image_accuracy, train_label_accuracy = run_epoch(model, train_loader, criterion, optimizer, epoch, mode='train')
        val_loss, val_image_accuracy, val_label_accuracy = run_epoch(model, val_loader, criterion, optimizer=None, epoch=epoch, mode='validation')

        history["train_loss"].append(train_loss)
        history["train_image_acc"].append(train_image_accuracy)
        history["train_label_acc"].append(train_label_accuracy)
        history["val_loss"].append(val_loss)
        history["val_image_acc"].append(val_image_accuracy)
        history["val_label_acc"].append(val_label_accuracy)

        print(f"Fine-tune Train Loss: {train_loss:.4f}, Train Image Accuracy: {train_image_accuracy:.4f}, Train Label Accuracy: {train_label_accuracy:.4f}")
        print(f"Fine-tune Validation Loss: {val_loss:.4f}, Validation Image Accuracy: {val_image_accuracy:.4f}, Validation Label Accuracy: {val_label_accuracy:.4f}")

        scheduler.step(val_loss)

        if val_label_accuracy > best_val_accuracy:
            best_val_accuracy = val_label_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at fine-tuning epoch {epoch+1}")

    plot_training_stats(history)
    print("Training complete.")

def plot_training_stats(history):
    '''
    Plots training data at the end of training
    '''
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(15, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker='o', color='red', linestyle='-')
    plt.plot(epochs, history["val_loss"], label="Validation Loss", marker='o', color='blue', linestyle='--')
    plt.plot(epochs, history["train_image_acc"], label="Train Image Accuracy", marker='x', color='green', linestyle='-.')
    plt.plot(epochs, history["val_image_acc"], label="Validation Image Accuracy", marker='x', color='orange', linestyle=':')
    plt.plot(epochs, history["train_label_acc"], label="Train Label Accuracy", marker='s', color='purple', linestyle='-')
    plt.plot(epochs, history["val_label_acc"], label="Validation Label Accuracy", marker='s', color='brown', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.title("Training and Validation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.show()

def custom_split(dataset):
    '''
    Gets one sample from each class for testing, puts the rest for training
    '''
    unique_labels = dataset.labels_df['target'].unique()
    
    train_indices = []
    test_indices = []
    
    for label in unique_labels:
        label_indices = dataset.labels_df[dataset.labels_df['target'] == label].index.tolist()
        test_index = np.random.choice(label_indices)
        test_indices.append(test_index)
        
        train_indices.extend([idx for idx in label_indices if idx != test_index])
    
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    return train_subset, test_subset

def stratified_split(dataset, test_size=0.2):
    labels = dataset.labels_df['labels']
    label_counts = dataset.labels_df['labels'].apply(lambda x: eval(x)[0]).value_counts()

    print(label_counts)

    train_indices = []
    test_indices = []
    
    for label in labels.unique():
        class_indices = labels[labels == label].index.tolist()
        num_test_samples = int(len(class_indices) * test_size)
        
        np.random.shuffle(class_indices)
        
        test_class_indices = class_indices[:num_test_samples]
        train_class_indices = class_indices[num_test_samples:]
        
        test_indices.extend(test_class_indices)
        train_indices.extend(train_class_indices)
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, test_dataset

if __name__ == '__main__':
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

    full_dataset = FundusDataset('input/ocular-disease-recognition-odir5k/ODIR-5K/Training Images', 
                                 'input/ocular-disease-recognition-odir5k/full_df.csv', 
                                 transform=data_transform)

    train_dataset, val_dataset = custom_split(full_dataset)

    num_workers = os.cpu_count()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    model = FundusModel().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.model.fc.parameters(), lr=4e-04, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    train_model(model, train_loader, val_loader, criterion, optimizer)

    torch.save(model.state_dict(), 'final_fundus_model.pth')
