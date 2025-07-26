import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),             
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),     
            nn.Flatten(),              
            nn.Linear(1024, 64),  
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.seq(x)

dataset = torchvision.datasets.CIFAR10(
    root=r"../data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
dataloader = DataLoader(dataset, batch_size=1)

loss = nn.CrossEntropyLoss()
net = Net()
optim = torch.optim.SGD(net.parameters(), lr=0.001)

epoches = 10
for epoch in range(epoches):
    epoch_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        output = net(imgs)
        result_loss = loss(output, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        epoch_loss += result_loss.item()
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in dataloader:
            outputs = model(imgs)
            predicted = torch.max(outputs, 1)[1]
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

evaluate(net, dataloader)

# Epoch 1, Loss: 2.2655
# Epoch 2, Loss: 2.0242
# Epoch 3, Loss: 1.8810
# Epoch 4, Loss: 1.7348
# Epoch 5, Loss: 1.6324
# Epoch 6, Loss: 1.5488
# Epoch 7, Loss: 1.4765
# Epoch 8, Loss: 1.4091
# Epoch 9, Loss: 1.3459
# Epoch 10, Loss: 1.2848
# Accuracy: 53.83%
