import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

padding_value=1
stride_value=1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=stride_value, padding=padding_value)
        self.bn1   = nn.BatchNorm2d(num_features=8)

        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16, kernel_size=3,stride=stride_value, padding=padding_value)
        self.bn2   = nn.BatchNorm2d(num_features=16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  

        self.conv3 = nn.Conv2d(in_channels=16,out_channels=16, kernel_size=3,stride=stride_value, padding=padding_value)
        self.bn3 = nn.BatchNorm2d(num_features=16)

        self.conv4 = nn.Conv2d(in_channels=16,out_channels=28, kernel_size=3,stride=stride_value, padding=padding_value)
        self.bn4 = nn.BatchNorm2d(num_features=28)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1_1 = nn.Conv2d(in_channels=28, out_channels=10, kernel_size=1) # 1x1 conv to get 10 channels for 10 classes

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) # 1x28x28 -> 8x28x28
        
        x = F.relu(self.bn2(self.conv2(x))) # 8x28x28 -> 16x28x28        
        x = self.pool1(x)                   # 16x28x28 -> 16x14x14

        x = F.relu(self.bn3(self.conv3(x))) # 16x14x14 -> 16x14x14
        x = self.pool2(x)                   # 16x14x14 -> 16x7x7

        x = F.relu(self.bn4(self.conv4(x))) # 16x7x7 -> 32x7x7

        #1 x1 convolution to get 10 channels for 10 classes
        x = self.conv1_1(x)                # 32x7x7 -> 10x7x7
        
        # Global Average Pooling
        x = self.gap(x)                     # 10x7x7 -> 10x1x1

        # Flatten
        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)



def get_optimizer_and_scheduler(model, train_loader_len, EPOCHS,lr=0.01, momentum=0.9,):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.015, steps_per_epoch=train_loader_len, epochs=EPOCHS, anneal_strategy='cos',pct_start=0.2,div_factor=10.0,final_div_factor=100.0 )
    return optimizer, scheduler