
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, n_classes, in_ch=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16,   32, 3, padding=1)
        self.conv3 = nn.Conv2d(32,   64, 3, padding=1)

        self.bn1, self.bn2, self.bn3 = nn.BatchNorm2d(16), nn.BatchNorm2d(32), nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.gap   = nn.AdaptiveAvgPool2d((1, 1))  
        self.drop  = nn.Dropout(0.3)

        self.fc1 = nn.Linear(64, 128)              
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x)                            
        x = x.view(x.size(0), -1)                   
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)
