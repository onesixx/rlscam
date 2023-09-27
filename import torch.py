import torch
import torch.nn as nn
import torch.optim as optim

# Define the perceptron model
class Perceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x

# Initialize the model
input_size = 2  # Number of input features
output_size = 1  # Number of output classes
model = Perceptron(input_size, output_size)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic gradient descent optimizer

# Generate some training data
x_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_train = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_train)
    
    # Compute the loss
    loss = criterion(outputs, y_train)
    
    # Backward pass and weight update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Test the model
x_test = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_test = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)
with torch.no_grad():
    outputs = model(x_test)
    predicted = (outputs > 0).float()
    accuracy = (predicted == y_test).float().mean()
    print(f"Accuracy: {accuracy.item():.4f}")