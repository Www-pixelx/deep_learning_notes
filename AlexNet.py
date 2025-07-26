import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq =nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
            )
        
    def forward(self, input):
        output = self.seq(input)
        return output
    
model = AlexNet()
X = torch.randn(1, 1, 224, 224)
for layer in model.seq:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)

# Conv2d output shape:	 torch.Size([1, 96, 54, 54])
# ReLU output shape:	 torch.Size([1, 96, 54, 54])
# MaxPool2d output shape:	 torch.Size([1, 96, 26, 26])
# Conv2d output shape:	 torch.Size([1, 256, 26, 26])
# ReLU output shape:	 torch.Size([1, 256, 26, 26])
# MaxPool2d output shape:	 torch.Size([1, 256, 12, 12])
# Conv2d output shape:	 torch.Size([1, 384, 12, 12])
# ReLU output shape:	 torch.Size([1, 384, 12, 12])
# Conv2d output shape:	 torch.Size([1, 384, 12, 12])
# ReLU output shape:	 torch.Size([1, 384, 12, 12])
# Conv2d output shape:	 torch.Size([1, 256, 12, 12])
# ReLU output shape:	 torch.Size([1, 256, 12, 12])
# MaxPool2d output shape:	 torch.Size([1, 256, 5, 5])
# Flatten output shape:	 torch.Size([1, 6400])
# Linear output shape:	 torch.Size([1, 4096])
# ReLU output shape:	 torch.Size([1, 4096])
# Dropout output shape:	 torch.Size([1, 4096])
# Linear output shape:	 torch.Size([1, 4096])
# ReLU output shape:	 torch.Size([1, 4096])
# Dropout output shape:	 torch.Size([1, 4096])
# Linear output shape:	 torch.Size([1, 10])