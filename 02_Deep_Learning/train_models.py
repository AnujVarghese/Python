"""
Deep Learning Training Script: ANN & CNN Models using PyTorch
- ANN: Industrial Sensor Machine Failure Diagnostics (Tabular)
- CNN: Visual Surface Defect Image Classification (64x64 Images)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. ANN MODULE: SENSOR VITALS DIAGNOSTICS
# -------------------------------------------------------------

class IndustrialVitalsANN(nn.Module):
    """Artificial Neural Network for Tabular Sensor Diagnostics."""
    def __init__(self, input_dim=5, num_classes=3):
        super(IndustrialVitalsANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def generate_sensor_data(n_samples=1200):
    """Generates synthetic sensor data (Vibration, Temp, Pressure, RPM, Noise)."""
    np.random.seed(42)
    vibration = np.random.uniform(0.1, 5.0, n_samples)
    temperature = np.random.uniform(30.0, 110.0, n_samples)
    pressure = np.random.uniform(1.0, 10.0, n_samples)
    rpm = np.random.uniform(800, 3500, n_samples)
    noise_db = np.random.uniform(40, 100, n_samples)

    # Risk score calculation
    risk_score = (vibration * 1.5) + (temperature * 0.05) + (pressure * 0.8) + (noise_db * 0.03) + (rpm / 1000.0)
    
    # Class labels: 0=Normal, 1=Warning, 2=Critical
    labels = np.zeros(n_samples, dtype=int)
    labels[risk_score >= 10.5] = 1
    labels[risk_score >= 13.5] = 2

    X = np.column_stack([vibration, temperature, pressure, rpm, noise_db])
    return X, labels

def train_ann_model():
    print("=== Training ANN (Machine Sensor Vitals) ===")
    X, y = generate_sensor_data(1200)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    train_tensor_x = torch.FloatTensor(X_train)
    train_tensor_y = torch.LongTensor(y_train)
    test_tensor_x = torch.FloatTensor(X_test)
    test_tensor_y = torch.LongTensor(y_test)

    train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = IndustrialVitalsANN(input_dim=5, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

    epochs = 30
    train_losses, train_accs = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

    # Save ANN model & scaler
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'ann_vitals_model.pth'))
    import joblib
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'ann_scaler.joblib'))
    
    # Save training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(train_losses, label='Loss', color='crimson')
    ax1.set_title('ANN Training Loss')
    ax1.legend()
    ax2.plot(train_accs, label='Accuracy', color='royalblue')
    ax2.set_title('ANN Training Accuracy')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'ann_training_curves.png'))
    plt.close()
    
    print(f"ANN Training Complete! Final Epoch Acc: {train_accs[-1]:.4f}")

# -------------------------------------------------------------
# 2. CNN MODULE: SURFACE DEFECT INSPECTION
# -------------------------------------------------------------

class DefectCNN(nn.Module):
    """Convolutional Neural Network for Surface Defect Detection."""
    def __init__(self, num_classes=4):
        super(DefectCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def generate_synthetic_defect_image(label, img_size=(64, 64)):
    """Generates synthetic surface defect images (0=Smooth, 1=Scratch, 2=Crack, 3=Stain)."""
    img = Image.new('RGB', img_size, color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    
    # Add noise texture
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 10, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    if label == 1: # Scratch
        draw.line([np.random.randint(5, 20), np.random.randint(5, 20), np.random.randint(40, 60), np.random.randint(40, 60)], fill=(30, 30, 30), width=2)
    elif label == 2: # Crack
        points = [(10, 10), (25, 30), (35, 25), (55, 50)]
        draw.line(points, fill=(10, 10, 10), width=3)
    elif label == 3: # Stain
        cx, cy = np.random.randint(20, 44), np.random.randint(20, 44)
        r = np.random.randint(8, 15)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(100, 80, 60))

    return img

class SyntheticDefectDataset(Dataset):
    def __init__(self, n_samples=800, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        
        for _ in range(n_samples):
            lbl = np.random.choice([0, 1, 2, 3])
            img = generate_synthetic_defect_image(lbl)
            self.samples.append(img)
            self.labels.append(lbl)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]
        lbl = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, lbl

def train_cnn_model():
    print("\n=== Training CNN (Surface Defect Inspection) ===")
    
    # Data Augmentation & Normalization Transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = SyntheticDefectDataset(n_samples=800, transform=train_transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DefectCNN(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    epochs = 20
    cnn_losses, cnn_accs = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        cnn_losses.append(epoch_loss)
        cnn_accs.append(epoch_acc)

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'cnn_defect_model.pth'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(cnn_losses, label='Loss', color='darkorange')
    ax1.set_title('CNN Training Loss')
    ax1.legend()
    ax2.plot(cnn_accs, label='Accuracy', color='seagreen')
    ax2.set_title('CNN Training Accuracy')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'cnn_training_curves.png'))
    plt.close()

    print(f"CNN Training Complete! Final Epoch Acc: {cnn_accs[-1]:.4f}")

if __name__ == '__main__':
    train_ann_model()
    train_cnn_model()
