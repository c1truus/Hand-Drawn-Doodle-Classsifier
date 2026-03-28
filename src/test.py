import torch
import numpy as np
from pathlib import Path
from train import DigitClassifier, load_mnist_split

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DigitClassifier().to(device)
model.load_state_dict(torch.load('../models/mnist_classifier.pth'))
model.eval()

# Load test data
print("Loading test data...")
test_images, test_labels = load_mnist_split('test')
test_tensor = torch.FloatTensor(test_images).to(device)
test_labels_tensor = torch.LongTensor(test_labels).to(device)

# Evaluate
correct = 0
with torch.no_grad():
    outputs = model(test_tensor)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == test_labels_tensor).sum().item()

accuracy = 100 * correct / len(test_labels)
print(f"Test Accuracy: {accuracy:.2f}%")