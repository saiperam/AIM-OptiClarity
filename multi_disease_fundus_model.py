import torch, pandas # type: ignore
from PIL import Image # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore 
from torchvision.models import resnet50, ResNet50_Weights # type: ignore
from torchvision.models import efficientnet_b3
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms # type: ignore
from pathlib import Path

class ImageDataset(Dataset): 
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform 

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform: 
            image = self.transform(image) # no need to convert to tensor, transforms.ToTensor() handles that
        return image, torch.tensor(self.labels[index], dtype=torch.float32)


def compute_dataset_statistics(image_paths, labels):
    preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor() # turns input into tensor and also divides each pixel value by 255
        ])

    image_dataset = ImageDataset(image_paths, labels, transform=preprocess)
    image_dataloader = DataLoader(image_dataset, batch_size=32, num_workers=NUM_WORKERS)
    mean, std = torch.zeros(3), torch.zeros(3)
    for image_batch, _ in image_dataloader:
        batch_size = image_batch.size(0)
        # image_batch current shape is (batch_size, num_channels, height, width) due to transforms.toTensor()
        # need to reshape it to (batch_size, num_channels, num_pixels) (num_pixels = height * width)
        curr_batch = image_batch.reshape(batch_size, image_batch.size(1), -1)
        # .mean(2) / .std(2) takes mean/std for each image in batch --> last dimension gets collapsed
        # .sum(0) sums means of all images across batch --> collapses first dimension
        # results are just 1D tensors of length 3 representing mean/std across all images in the batch for each of the 3 channels (red, green, blue)
        mean += curr_batch.mean(2).sum(0)
        std += curr_batch.std(2).sum(0)
    
    # average means and stds across all batches so that CNN works with consisten mean/std 
    # for ALL batches. Using different mean and std for each batch introduces inconsistency in the model, negatively affecting its learning
    mean /= len(image_paths)
    std /= len(image_paths)
    return mean, std


def create_train_val_dataloaders(image_paths, labels, transform=None):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist())
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist())
    ])
    train_size = int(0.8*len(image_paths))
    val_size = len(image_paths) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset=ImageDataset(image_paths, labels, transform),
        lengths=[train_size, val_size]
    )
    train_dataset.dataset = ImageDataset(image_paths, labels, train_transform)
    val_dataset.dataset = ImageDataset(image_paths, labels, val_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    return train_dataloader, val_dataloader


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 4
NUM_EPOCHS = 20
NUM_CLASSES = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3

df = pandas.read_csv(Path("full_df.csv"))
image_paths = [Path("ODIR-5K/ODIR-5K/Training_Images") / filename for filename in df["filename"]]
labels = df.iloc[:, 7:15].values

#resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet = efficientnet_b3(pretrained=True)
# Replace the last fully-connected layer
num_features = resnet.classifier[1].in_features
resnet.classifier[1] = torch.nn.Linear(in_features=num_features, out_features=NUM_CLASSES)
for param in resnet.classifier.parameters():
    param.requires_grad = True
resnet.to(DEVICE)
optimizer = torch.optim.AdamW(list(resnet.classifier.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
# dealing with multi-label classification --> use BCEWithLogitsLoss, handles each binary loss independently
# and removes need for applying softmax before passing into loss function (takes in logits directly)
bce_loss = torch.nn.BCEWithLogitsLoss()

mean, std = compute_dataset_statistics(image_paths, labels)
image_preprocessing = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist())
])
train_dataloader, val_dataloader = create_train_val_dataloaders(image_paths, labels, transform=image_preprocessing)


# Apply transfer learning to the model
best_val_accuracy = float('-inf')
for epoch in range(NUM_EPOCHS):
    resnet.train()
    epoch_loss = 0
    for train_batch_images, train_batch_labels in train_dataloader:
        optimizer.zero_grad()
        image_batch, label_batch = train_batch_images.to(DEVICE), train_batch_labels.to(DEVICE) 
        model_outputs = resnet(image_batch)
        loss = bce_loss(model_outputs, label_batch)
        
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * image_batch.size(0)

    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(image_paths)}")

    num_val_correct = 0
    num_val = 0
    for val_image, val_label in val_dataloader:
        image, label = val_image.to(DEVICE), val_label.to(DEVICE)
        res_out = resnet(image)
        output_probabilities = torch.nn.Sigmoid()(res_out)
        predictions = (output_probabilities >= 0.5).float()
        num_val_correct += (predictions == label).float().sum().item()
        num_val += label.numel()

    val_accuracy = num_val_correct / num_val
    print(f"Epoch {epoch+1} test accuracy: {val_accuracy}")
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(resnet.state_dict(), Path("best_resnet50_weights.pth"))

    scheduler.step()

print(f"best test accuracy: {val_accuracy}")





