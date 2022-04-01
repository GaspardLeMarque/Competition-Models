import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score

sns.set(style="darkgrid", font_scale=1.4)

# Generate dataset and plot it

X, y = make_moons(n_samples=10000, random_state=42, noise=0.1)

plt.figure(figsize=(16, 10))
plt.title("Dataset")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
plt.show()

# Split the sample

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)

# Create tensors

X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)
X_val_t = torch.from_numpy(X_val)
y_val_t = torch.from_numpy(y_val)

# Form batches

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t) # testset
train_dataloader = DataLoader(train_dataset, batch_size=128)
val_dataloader = DataLoader(val_dataset, batch_size=128) # testloader

# Write nn.Linear module from scratch

class LinearRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features))                         
        self.bias = bias
        if bias:
            # my code
            self.bias_term = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # my code
        x = x @ self.weights.t()
        if self.bias:
            # my code
            x += self.bias_term
        return x

# Add optimizer and loss function

linear_regression = LinearRegression(2, 1)
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(linear_regression.parameters(), lr=0.05)

# Fitting cycle

tol = 1e-3
losses = []
max_epochs = 100
prev_weights = torch.zeros_like(linear_regression.weights)
stop_it = False
data = train_dataloader, val_dataloader
# Several times iterate through dataset
for epoch in range(max_epochs): 
# Take batches for SGD
    for it, (X_batch, y_batch) in enumerate(train_dataloader):
# Zeroing model's gradients
        optimizer.zero_grad()
# Get logits from the model      
        outp = linear_regression.forward(X_batch.float()) # Use linear_regression to get outputs
        loss = loss_function(outp.flatten(), y_batch.float()) # Compute loss
# Compute gradients
        loss.backward()
        losses.append(loss.detach().flatten()[0])
# Make a gradient step
        optimizer.step()
        probabilities = 1 - losses[it] # Compute probabilities
        preds = (probabilities > 0.5).type(torch.long)
        batch_acc = (preds.flatten() == y_batch).type(torch.float32).sum() / y_batch.size(0)
        
        if (it + epoch * len(train_dataloader)) % 100 == 0:
            print(f"Iteration: {it + epoch * len(train_dataloader)}\nBatch accuracy: {batch_acc}")
        current_weights = linear_regression.weights.detach().clone()
        if (prev_weights - current_weights).abs().max() < tol:
            print(f"\nIteration: {it + epoch * len(train_dataloader)}.Convergence. Stopping iterations.")
            stop_it = True
            break
        prev_weights = current_weights
    if stop_it:
        break

# Visualize the results
plt.figure(figsize=(12, 8))
plt.plot(range(len(losses)), losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

sns.set(style="white")

xx, yy = np.mgrid[-1.5:2.5:.01, -1.:1.5:.01]
grid = np.c_[xx.ravel(), yy.ravel()]
batch = torch.from_numpy(grid).type(torch.float32)
with torch.no_grad():
    probs = torch.sigmoid(linear_regression(batch).reshape(xx.shape))
    probs = probs.numpy().reshape(xx.shape)

f, ax = plt.subplots(figsize=(16, 10))
ax.set_title("Decision boundary", fontsize=14)
contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                      vmin=0, vmax=1)
ax_c = f.colorbar(contour)
ax_c.set_label("$P(y = 1)$")
ax_c.set_ticks([0, .25, .5, .75, 1])

ax.scatter(X[100:,0], X[100:, 1], c=y[100:], s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

ax.set(xlabel="$X_1$", ylabel="$X_2$")
plt.show()

# Make prediction with test

@torch.no_grad()
def predict(dataloader, model):
    model.eval()
    predictions = np.array([])
    for x_batch, _ in dataloader:
    	# Use sigmoid transformation to calculate probabilities and sort predictions to classes by condition (> 0.5)
        preds = torch.sigmoid(model(x_batch.float())) > 0.5 
        predictions = np.hstack((predictions, preds.numpy().flatten()))
    return predictions.flatten()

# Compute total accuracy
print('Accuracy of the network:', accuracy_score(y_val_t, predict(val_dataloader, linear_regression)))    

# Interesting that model without sigmoid [model(x_batch.float())] performs even better (Accuracy = 0.87)    
