import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import time

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load your clean MNIST data
def load_mnist_split(split='train'):
    """Load numpy files from train/valid/test splits"""
    proj_root = Path.cwd().parent
    print(proj_root)
    data_dir = proj_root / 'data'
    images_dir = data_dir / split / 'images'
    labels_dir = data_dir / split / 'labels'
    
    images = []
    labels = []
    
    # Get all image files
    img_files = sorted(images_dir.glob('*.npy'))
    
    for img_file in img_files:
        img = np.load(img_file).astype(np.float32)  # Convert to float32
        label = np.load(labels_dir / img_file.name)
        images.append(img)
        labels.append(label)
    
    # Stack into arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Normalize images to [0, 1] and flatten
    images = images.reshape(len(images), -1) / 255.0
    
    return images, labels

print("Loading data...")
train_images, train_labels = load_mnist_split('train')
valid_images, valid_labels = load_mnist_split('valid')
print(f"Train: {train_images.shape}, Valid: {valid_images.shape}")

# Convert to PyTorch tensors
train_tensor = torch.FloatTensor(train_images).to(device)
train_labels_tensor = torch.LongTensor(train_labels).to(device)
valid_tensor = torch.FloatTensor(valid_images).to(device)
valid_labels_tensor = torch.LongTensor(valid_labels).to(device)

# Create DataLoaders
batch_size = 64
train_dataset = TensorDataset(train_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataset = TensorDataset(valid_tensor, valid_labels_tensor)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

# Define the network
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 10)
        )
    
    def forward(self, x):
        return self.network(x)

# Initialize model, loss, optimizer
model = DigitClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
train_losses = []
valid_accuracies = []

print("\nStarting training...")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
    
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # Validation phase
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in valid_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    valid_accuracies.append(accuracy)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

print("\nTraining complete!")

# Save the model
model_path = '../models/mnist_classifier.pth'
Path('../models').mkdir(exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Save training history
import json
history = {
    'train_losses': train_losses,
    'valid_accuracies': valid_accuracies,
    'final_accuracy': accuracy
}
with open('../models/training_history.json', 'w') as f:
    json.dump(history, f)

# Simple visualization of training progress
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses)
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

ax2.plot(valid_accuracies)
ax2.set_title('Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('../images/training_curves.png')
plt.show()

print(f"\nFinal Validation Accuracy: {accuracy:.2f}%")