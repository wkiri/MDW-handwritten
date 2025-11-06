#!/usr/bin/env python3
# Kiri Wagstaff
# July 25, 2025
# Train a basic CNN for MNIST digits using pytorch
# Reference: https://medium.com/@ponchanon.rone/building-a-cnn-for-handwritten-digit-recognition-with-pytorch-a-step-by-step-guide-9df1dcb1092d

import sys
import os
import torch
from torch import load
from torch import save
from torch.optim import Adam
from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset

# Create a training data loader for MNIST
def get_data_loader(batch_size=32):
    train_data = datasets.MNIST(root='data', train=True, download=True,
                                transform=ToTensor())
    return DataLoader(train_data, batch_size=batch_size, shuffle=True)


'''
# Define the neural network (CNN): 3 convolutional layers + output fully connected
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # Output: 26x26
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3), # Output: 24x24
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), # Output: 22x22
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, 10)       # Fully connected layer for 10 classes
        )

    def forward(self, x):
        return self.model(x)

# Define the neural network (CNN): 5 convolutional layers + output fully connected
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # Output: 26x26
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3), # Output: 24x24
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), # Output: 22x22
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), # Output: 20x20
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), # Output: 18x18
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 18 * 18, 10)       # Fully connected layer for 10 classes
        )

    def forward(self, x):
        return self.model(x)
'''

# Define the neural network (VGG-like): 5 convolutional layers + output fully connected
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)
    

# Train a CNN on MNIST data for n_epochs and save it to model_file
# Use the Adam optimizer with specified learning rate (lr)
# and cross-entropy loss
def train_model(n_epochs, model_file, device='mps', lr=1e-3):
    loss_history = []  # List to track losses
    model = ImageClassifier().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()
    data_loader = get_data_loader()
    model_base = model_file[0:model_file.find('.')]

    for epoch in range(n_epochs):
        total_loss = 0
        # Train in batches
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        epoch_loss = total_loss/len(data_loader)
        loss_history.append(epoch_loss)  # Append epoch loss to history
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss}')

        # Save checkpoint model every 10 epochs
        if (epoch + 1) % 10 == 0:
            fn = f'{model_base}-e{epoch+1}.pt'
            with open(fn, 'wb') as f:
                save(model.state_dict(), f)
            print(f'Saved checkpoint model to {fn}.')

    # Save just the weights
    #torch.save(model.state_dict(), 'model_state.pt')
    # Save the full model
    with open(model_file, 'wb') as f: 
        save(model.state_dict(), f)
    print(f'Training complete! Model saved as {model_file}.')

    return model


# Train a CNN on MNIST data for n_epochs and save it to model_file
# Evaluate it on the MNIST test set
# 'mps' is a dedicated GPU found in M-series chips (like Mom's laptop)
# For this task, it is about 3x faster than 'cpu'
def main(model_file, n_epochs, device='mps'):

    if not os.path.exists(model_file):
        # Can also try cuda if I get it installed
        # On Mom's laptop, it takes about 1.5 minutes per epoch to train with cpu
        train_model(n_epochs=n_epochs, model_file=model_file, device=device)

    # Load the model back in
    model = ImageClassifier().to(device=device)
    model.load_state_dict(load(model_file, map_location=device))
    model.eval() # Set it to evaluation mode (disable dropout and batch norm)

    # Load the test data from QMNIST so we get HSF etc. metadata
    qtest = datasets.QMNIST(root='./data', train=False, compat=False,
                            download=True, transform=ToTensor())
    # Only use the first 10k for testing
    qtest = Subset(qtest, torch.arange(10000))
    test_loader = DataLoader(qtest, batch_size=64, shuffle=False)

    n_correct = 0
    n_census_correct = 0
    n_high_school_correct = 0
    n_census = 0
    n_high_school = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculation during evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Max returns values and indices
            _, predicted = torch.max(outputs.data, 1)
            # Get the HSF for each item so we can check for bias
            hsf = labels[:,1]
            labels = labels[:,0] # just grab the class label
            total += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            census = torch.where(hsf != 4)[0]
            n_census += len(census)
            high_school = torch.where(hsf == 4)[0]
            n_high_school += len(high_school)
            n_census_correct += (predicted[census] == labels[census]).sum().item()
            n_high_school_correct += (predicted[high_school] == labels[high_school]).sum().item()

    accuracy = 100 * n_correct / total
    print(f'Accuracy of the CNN on MNIST test images: {accuracy:.4f}%')

    # Assess adult vs. high school (HSF 4) error rates
    acc = 100 * n_census_correct / n_census
    print(f'  Census employee digits (n={n_census}): {acc:.4f}%')
    acc = 100 * n_high_school_correct / n_high_school
    print(f'  High school student digits (n={n_high_school}): {acc:.4f}%')
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('model_file', 
                        help='Save the trained model here')
    parser.add_argument('n_epochs', type=int, default=100,
                        help='Number of epochs to train')

    args = parser.parse_args()
    main(**vars(args))

