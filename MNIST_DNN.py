import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import TensorDataset, DataLoader

sns.set(style="darkgrid", font_scale=1.4)

import os
import torchvision.transforms as tfs 

# Import dataset and create batches

from torchvision.datasets import MNIST

data_tfs = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize((0.5), (0.5))
])

# Set up for train and test with increased batch size

root = './'
train_dataset = MNIST(root, train=True,  transform=data_tfs, download=True)
val_dataset  = MNIST(root, train=False, transform=data_tfs, download=True)

train_dataloader =  torch.utils.data.DataLoader(train_dataset, batch_size=512,
                                          shuffle=True, num_workers=2)
valid_dataloader =  torch.utils.data.DataLoader(val_dataset, batch_size=512,
                                         shuffle=False, num_workers=2)

# Use Dense Neural Network with Exponential Linear Unit as an activation function

class Identical(nn.Module):
    def forward(self, x):
        return x
      
# D_in - input size (number of features)
# H - size of hidden layers
# D_out - output size (number of classes)

D_in, H, D_out = 784, 128, 10

activation = nn.ELU

model = nn.Sequential( 
    nn.Flatten(), 
    nn.Linear(D_in, H), 
    activation(), 
    nn.Linear(H, H), 
    activation(), 
    nn.Linear(H, H), 
    activation(), 
    nn.Linear(H, D_out), 
)      

# Select loss function and create condition for device

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

loaders = {"train": train_dataloader, "valid": valid_dataloader}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Fitting cycle

max_epochs = 10
accuracy = {"train": [], "valid": []}
# Send model to a device
model.to(device)
# Iterate through dataset 
for epoch in range(max_epochs):
    for k, dataloader in loaders.items():
        epoch_correct = 0
        epoch_all = 0
        for x_batch, y_batch in dataloader:
            # Send batches to a device 
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if k == "train":
                 # Set model to ``train`` mode and calculate outputs
                 model.train()
                 # Zeroing gradient
                 optimizer.zero_grad()
                 outp = model(x_batch)
            else:
                 # Set model to ``eval`` mode and calculate outputs
                 model.eval()
                 with torch.no_grad():
                   outp = model(x_batch)
            preds = outp.argmax(-1)
            correct = (preds == y_batch).sum()
            all = y_batch.size(0) 
            epoch_correct += correct.item()
            epoch_all += all
            if k == "train":
                loss = criterion(outp, y_batch)
                # Calculate gradients and make a step of your optimizer
                loss.backward()
                optimizer.step()
        if k == "train":
            print(f"Epoch: {epoch+1}")
        print(f"Loader: {k}. Accuracy: {epoch_correct/epoch_all}")
        accuracy[k].append(epoch_correct/epoch_all)

# Save accuracy for ELU activation
elu_accuracy = accuracy["valid"]

# Fitting cycle for the other activation functions

max_epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
def test_activation_function(activation):
    # Create nn
    D_in, H, D_out = 784, 128, 10

    model = nn.Sequential( 
          nn.Flatten(), 
          nn.Linear(D_in, H), 
          activation(), 
          nn.Linear(H, H), 
          activation(), 
          nn.Linear(H, H), 
          activation(), 
          nn.Linear(H, D_out), 
      )  
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    loaders = {"train": train_dataloader, "valid": valid_dataloader}
    
    # Fit the network
    
    # Send model to device
    model.to(device)
    # Iterate through dataset 
    for epoch in range(max_epochs):
        for k, dataloader in loaders.items():
            epoch_correct = 0
            epoch_all = 0
            for x_batch, y_batch in dataloader:
                # Send batches to device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                if k == "train":
                    # Set model to ``train`` mode and calculate outputs
                    model.train()
                    # Zeroing gradient
                    optimizer.zero_grad()
                    outp = model(x_batch)
                else:
                    # Set model to ``eval`` mode and calculate outputs
                    model.eval()
                    with torch.no_grad():
                      outp = model(x_batch)
                preds = outp.argmax(-1)
                # My Code
                correct = (preds == y_batch).sum()
                # My Code
                all = y_batch.size(0) 
                epoch_correct += correct.item()
                epoch_all += all
                if k == "train":
                    loss = criterion(outp, y_batch)
                    # Calculate gradients and make a step of your optimizer
                    loss.backward()
                    optimizer.step()
            if k == "train":
                print(f"Epoch: {epoch+1}")
            print(f"Loader: {k}. Accuracy: {epoch_correct/epoch_all}")
            accuracy[k].append(epoch_correct/epoch_all)

# Plain (No activation function)

accuracy = {"train": [], "valid": []}
test_activation_function(Identical)
plain_accuracy = accuracy["valid"]

# ReLU

accuracy = {"train": [], "valid": []}
test_activation_function(nn.ReLU)
relu_accuracy = accuracy["valid"]

# Leaky ReLU

accuracy = {"train": [], "valid": []}
test_activation_function(nn.LeakyReLU)
leaky_relu_accuracy = accuracy["valid"]

# Display all activations

print(f'Plain accuracy: {plain_accuracy}')
print(f'ReLU accuracy: {relu_accuracy}')
print(f'Leaky ReLU accuracy: {leaky_relu_accuracy}')
print(f'ELU accuracy: {elu_accuracy}')

# Build a plot to compare calculated accuracies

plt.figure(figsize=(16, 10))
plt.title("Valid accuracy")
plt.plot(range(max_epochs), plain_accuracy, label="No activation", linewidth=2)
plt.plot(range(max_epochs), relu_accuracy, label="ReLU activation", linewidth=2)
plt.plot(range(max_epochs), leaky_relu_accuracy, label="LeakyReLU activation", linewidth=2)
plt.plot(range(max_epochs), elu_accuracy, label="ELU activation", linewidth=2)
plt.legend()
plt.xlabel("Epoch")
plt.show()
