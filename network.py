import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SelfDrivingModel(nn.Module):
    def __init__(self):
        super(SelfDrivingModel, self).__init__()
        # Load the pre-trained ResNet-101 model (pre-trained on ImageNet)
        resnet101 = models.resnet101(pretrained=True)

        #Adding pre-trained ResNet-101 layer in the initial layers
        self.conv1 = resnet101.conv1
        self.bn1 = resnet101.bn1
        self.maxpool = resnet101.maxpool
        self.layer1 = resnet101.layer1
        self.layer2 = resnet101.layer2
        self.layer3 = resnet101.layer3
        self.layer4 = resnet101.layer4

        self.fc1 = nn.Linear(2048, 1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(self.maxpool(x))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc5(x)
        y = torch.atan(x) * 2  # scale the atan output
        return y

# Create an instance of the model
model = SelfDrivingModel()
