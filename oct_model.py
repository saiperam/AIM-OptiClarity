import torch, timm
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models import efficientnet_b3
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import pathlib 


class OCTImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        super(OCTImageDataset, self).__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(self.labels[index], dtype=torch.long)
    
class Ranker(nn.Module):
    def __init__(self, num_classes):
        super(Ranker, self).__init__()
        self.fc_layer = nn.Linear(in_features=3*num_classes, out_features=num_classes) 
    def forward(self, vit_output, inc_v3_output, resnet_output):
        return self.fc_layer(torch.cat([vit_output, inc_v3_output, resnet_output], dim=1))

def compute_dataset_statistics(image_paths, labels):
    stat_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    oct_dataset = OCTImageDataset(image_paths, labels, transform=stat_transform)
    oct_dataloader = DataLoader(oct_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    mean, std = torch.zeros(3).to(DEVICE), torch.zeros(3).to(DEVICE)
    for image_batch, label_batch in oct_dataloader:
        images, labels = image_batch.to(DEVICE), label_batch.to(DEVICE)
        images = images.reshape(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    return mean / len(image_paths), std / len(image_paths)

def load_train_test_dataloaders(train_image_paths, train_labels, test_image_paths, test_labels, transform=None):
    train_dataset = OCTImageDataset(train_image_paths, train_labels, transform)
    test_dataset = OCTImageDataset(test_image_paths, test_labels, transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    return train_loader, test_dataloader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 8
NUM_EPOCHS = 20

eff_net = efficientnet_b3(pretrained=True)
inc_v3 = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

for param in eff_net.parameters(): 
    param.requires_grad = True
for param in inc_v3.parameters():
    param.requires_grad = True
for param in resnet.parameters(): 
    param.requires_grad = True

eff_net.classifier[1] = nn.Linear(eff_net.classifier[1].in_features, NUM_CLASSES)
#vit.head = nn.Linear(in_features=vit.head.in_features, out_features=NUM_CLASSES)
inc_v3.fc = nn.Linear(in_features=inc_v3.fc.in_features, out_features=NUM_CLASSES)
resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=NUM_CLASSES)

image_dir = pathlib.Path("RetinalOCT_Dataset/RetinalOCT_Dataset")
train_dir, val_dir, test_dir = image_dir / "train", image_dir / "val", image_dir / "test" 
train_image_paths, train_labels = [], []
name_to_label = {
    "NORMAL": 0, "MH": 1, "DRUSEN": 2, 
    "DR": 3, "DME": 4, "CSR": 5, "CNV": 6, "AMD": 7
}


for disease_dir in train_dir.iterdir():
    label = name_to_label[pathlib.PurePath(disease_dir).name]
    for file_path in disease_dir.iterdir():
        train_image_paths.append(file_path)
        train_labels.append(label)
for disease_dir in val_dir.iterdir():
    label = name_to_label[pathlib.PurePath(disease_dir).name]
    for file_path in disease_dir.iterdir():
        train_image_paths.append(file_path)
        train_labels.append(label)
test_image_paths, test_labels = [], []
for disease_dir in test_dir.iterdir():
    label = name_to_label[pathlib.PurePath(disease_dir).name]
    for file_path in disease_dir.iterdir():
        test_image_paths.append(file_path)
        test_labels.append(label)

mean, std = compute_dataset_statistics(train_image_paths, train_labels)
preprocessing = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist())
    ])
ranker, cross_entropy_loss = Ranker(NUM_CLASSES), nn.CrossEntropyLoss()
train_loader, test_loader = load_train_test_dataloaders(train_image_paths, train_labels, test_image_paths, test_labels, transform=preprocessing)
trainable_params = list(eff_net.parameters()) + list(inc_v3.parameters()) + list(resnet.parameters()) + list(ranker.parameters())
optimizer = Adam(trainable_params, lr=3e-4, weight_decay=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

inc_v3.to(DEVICE)
resnet.to(DEVICE)
eff_net.to(DEVICE)
ranker.to(DEVICE)

best_test_accuracy = float('-inf')
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    eff_net.train()
    inc_v3.train()
    resnet.train()
    ranker.train()
    for image_batch, label_batch in train_loader:
        images, labels = image_batch.to(DEVICE), label_batch.to(DEVICE)
        vit_output, inc_v3_output, resnet_output = eff_net(images), inc_v3(images).logits, resnet(images)
        output_logits = ranker(vit_output, inc_v3_output, resnet_output)
        loss = cross_entropy_loss(output_logits, labels)
        epoch_loss += loss.item() * image_batch.size(0)
        predicted = torch.argmax(output_logits, dim=1, keepdim=False).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Loss for epoch {epoch+1}: {epoch_loss / len(train_image_paths)}")

    eff_net.eval()
    inc_v3.eval()
    resnet.eval()
    ranker.eval()
    num_correct = 0
    num_test = 0
    with torch.no_grad():
        for image_batch, label_batch in test_loader:
            images, labels = image_batch.to(DEVICE), label_batch.to(DEVICE)
            vit_output, inc_v3_output, resnet_output = eff_net(images), inc_v3(images), resnet(images)
            predicted = torch.argmax(ranker(vit_output, inc_v3_output, resnet_output), dim=1, keepdim=False) # (batch_size, )
            num_correct += (predicted == labels).sum()
            num_test += predicted.numel()

    test_accuracy = num_correct / num_test
    print(f"TEST ACCURACY FOR EPOCH {epoch+1}: {test_accuracy}")
    if test_accuracy > best_test_accuracy :
        best_test_accuracy = test_accuracy
        torch.save(inc_v3.state_dict(), pathlib.Path("best_inception_weights.pth"))
        torch.save(resnet.state_dict(), pathlib.Path("best_resnet_weights.pth"))
        torch.save(eff_net.state_dict(), pathlib.Path("best_vit_weights.pth"))
        torch.save(ranker.state_dict(), pathlib.Path("best_ranker_weights.pth"))
    scheduler.step()

print("best test accuracy: " + str(best_test_accuracy))
