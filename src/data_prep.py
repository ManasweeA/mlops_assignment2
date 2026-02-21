import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import os

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_data():
    train_ds = datasets.ImageFolder('data/raw/training_set', transform=transform_train)
    test_ds = datasets.ImageFolder('data/raw/test_set', transform=transform_test)
    
    # Split train into train/val (80/10, test=10%)
    train_size = int(0.89 * len(train_ds))  # ~7120 train, ~880 val
    val_size = len(train_ds) - train_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])
    
    torch.save(train_ds, 'data/processed/train.pt')
    torch.save(val_ds, 'data/processed/val.pt')
    torch.save(test_ds, 'data/processed/test.pt')
    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    preprocess_data()
