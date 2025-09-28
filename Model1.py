import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import StepLR
class Net(nn.Module):
    def __init__(self):
        mypadding=1
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=mypadding)   # 1 → 16
        self.bn1 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=mypadding)  # 16 → 32
        self.bn2 = nn.BatchNorm2d(16)
        self.conv1x1_1 = nn.Conv2d(16, 8, kernel_size=1)         # shrink channels

        self.conv3 = nn.Conv2d(8, 32, kernel_size=3, padding=mypadding)  # back to 32
        self.bn3 = nn.BatchNorm2d(16)

        self.conv1x1_2 = nn.Conv2d(32, 28, kernel_size=1)         # refine features

        # self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=mypadding)  # back to 32
        # self.conv1x1_3 = nn.Conv2d(32, 32, kernel_size=1)         # refine features

        self.fc1 = nn.Linear(28*7*7, 10)  # after 2 poolings: 28→14→7

    def forward(self, x):
        # First block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # [batch, 16, 14, 14]

        # Second block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # [batch, 32, 7, 7]
        x = F.relu(self.conv1x1_1(x))          # [batch, 16, 7, 7]

        # Third block
        x = F.relu(self.conv3(x))              # [batch, 32, 7, 7]
        x = F.relu(self.conv1x1_2(x))          # [batch, 32, 7, 7]

        # # Fourth block
        # x = F.relu(self.conv4(x))              # [batch, 32, 7, 7]
        # x = F.relu(self.conv1x1_3(x))          # [batch, 32, 7, 7]

        # Flatten + FC
        x = torch.flatten(x, 1)                # [batch, 32*7*7]
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
    

def get_optimizer_and_scheduler(model, lr=0.01, momentum=0.9, step_size=6, gamma=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    return optimizer, scheduler