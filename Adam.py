import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Custom Adam Optimizer
class CustomAdam:
    def __init__(self, params, lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m_t = [torch.zeros_like(p) for p in self.params]
        self.v_t = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            self.m_t[i] = self.beta_1 * self.m_t[i] + (1 - self.beta_1) * p.grad
            self.v_t[i] = self.beta_2 * self.v_t[i] + (1 - self.beta_2) * p.grad ** 2

            m_hat = self.m_t[i] / (1 - self.beta_1 ** self.t)
            v_hat = self.v_t[i] / (1 - self.beta_2 ** self.t)

            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()



X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.int64))  
y_test = torch.from_numpy(y_test.astype(np.int64))    

n_input, n_features = X_train.shape

# Simple feedforward neural network for multi-class classification
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)  

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out) 
        return out


# Models
num_classes = 10  
model1 = SimpleNN(input_size=n_features, num_classes=num_classes)
model2 = SimpleNN(input_size=n_features, num_classes=num_classes)
model3 = SimpleNN(input_size=n_features, num_classes=num_classes)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizers
learning_rate = 0.09

optimizer1 = torch.optim.SGD(model1.parameters(), lr=learning_rate)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
optimizer3 = CustomAdam(model3.parameters(), lr=learning_rate)


# Training loop
n_epochs = 200
def train(model, optimizer, criterion, name):
    for epoch in range(n_epochs):
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Zero gradients, backward pass, and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{n_epochs}], Loss: {loss.item():.4f}')

    with torch.no_grad():
        predictions = model(X_test)
        _, predicted_classes = torch.max(predictions, 1) 
        accuracy = (predicted_classes == y_test).float().mean()
        print(f'Accuracy on test data for {name}: {accuracy.item() * 100:.2f}% \n')

    
print("SGD optimizer:")
train(model1, optimizer1, criterion, "SGD")
print("Adam optimizer:")
train(model2, optimizer2, criterion, "Adam")
print("Custom Adam optimizer:")
train(model3, optimizer3, criterion, "Custom Adam")
