import torch
import torch.nn as nn                                 #achieve all kinds of layers
import torch.nn.functional as F                       #achieve all kinds of functions
import torch.optim as optim                           #achieve all packages of optim algorithms
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()                   #init super class
        #init the type of layer
        self.conv1 = nn.Conv2d(1, 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        #define forward propagation
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        print("layer1 output:")
        print(x.size())
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        print("layer2 output:")
        print(x.size())
        x = x.view(-1, self.num_flat_features(x))       #reshape
        print("layer3 output:")
        print(x.size())
        x = F.relu(self.fc1(x))
        print("layer4 output:")
        print(x.size())
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        print("layer5 output:")
        print(x.size())
        x = F.log_softmax(x, dim = 1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# model = Net().to(device)
# print(model)

# input = torch.randn(1, 1, 32, 32)
# print("input size:")
# print(input.size())
# output = model(input)
# print("output:")
# print(output)
