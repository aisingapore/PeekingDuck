import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

print("GPU available:\n\n", torch.cuda.is_available())
print("CUDA version:\n\n", torch.version.cuda)
print("cudnn version:", torch.backends.cudnn.version())

# Define the model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Define the dataset and data loader
class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.randn((100, 10))
        self.targets = torch.randn((100, 1))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


dataset = MyDataset()
data_loader = DataLoader(dataset, batch_size=16)

# Initialize the model and move it to the GPU
model = MyModel().cuda()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(data_loader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}: loss={running_loss/len(data_loader)}")
