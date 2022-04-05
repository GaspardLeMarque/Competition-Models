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

# Set up for train and test with decreased batch size in comparison to DNN

root = './'
train_dataset = MNIST(root, train=True,  transform=data_tfs, download=True)
val_dataset  = MNIST(root, train=False, transform=data_tfs, download=True)

train_dataloader =  torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                          shuffle=True, num_workers=2)
valid_dataloader =  torch.utils.data.DataLoader(val_dataset, batch_size=128,
                                         shuffle=False, num_workers=2)

# Use modified LeNet with ReLU as an activation function 

class LeNet(nn.Module):
    def __init__(self):
        # Seed for reproducibility purposes
        torch.manual_seed(42)
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # 6 input channels, 16 output channels 
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2) 
        # Fully connected layer with 120 neurons
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        # Fully connected layer with 84 neurons
        self.fc2 = nn.Linear(120, 84)
        # Output layer with 10 neurons
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Apply layers created in __init__. 
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1) #x = x.view(-1, 5 * 5 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Select loss function, create condition for device and create dictionary for fitting cycle

device = 'cuda' if torch.cuda.is_available() else 'cpu'      
model = LeNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

loaders = {"train": train_dataloader, "valid": valid_dataloader}      

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
        
# Display LeNet accuracy

lenet_accuracy = accuracy["valid"]
print(lenet_accuracy[-1]) # almost 99%

# Show accuracy on the plot

plt.figure(figsize=(16, 10))
plt.title("Valid accuracy")
plt.plot(range(max_epochs), lenet_accuracy, label="LeNet", linewidth=2)
plt.legend()
plt.xlabel("Epoch")
plt.show()
