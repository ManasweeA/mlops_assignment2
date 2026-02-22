import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_prep import preprocess_data
mlflow.pytorch.autolog()

class CatsDogsCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    def forward(self, x): return self.conv(x)

with mlflow.start_run():
    train_ds, val_ds, _ = preprocess_data()
    train_loader = DataLoader(train_ds, 32, True)
    val_loader = DataLoader(val_ds, 32)
    
    model = CatsDogsCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    mlflow.log_params({"lr": 0.001, "epochs": 10})
    for epoch in range(10):
        model.train()
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        # Val metrics logged by autolog
        
    mlflow.pytorch.log_model(model, "model")
