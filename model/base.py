import torch
import torch.nn as nn
import torch.nn.functional as Func
import abc

class Model(abc.ABC):
    @abc.abstractmethod
    def pretrain(self, dataset, train_index, vali_index):
        print("Should never reach here")

    @abc.abstractmethod
    def train(self, dataset, train_index, vali_index):
        print("Should never reach here")

    @abc.abstractmethod
    def test(self, dataset, test_index):
        print("Should never reach here")

class CNNMnistBody(nn.Module):
    def __init__(self):
        super(CNNMnistBody, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

    def forward(self, x):
        x = Func.relu(Func.max_pool2d(self.conv1(x), 2))
        x = Func.relu(Func.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        return x

# for initialization only
class CNNMnistHead():
    def __init__(self):
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

class CNNMnist(nn.Module):
    def __init__(self, body=None, fc1=None, fc2=None, augment=False):
        super(CNNMnist, self).__init__()
        self.body = body
        self.fc1 = fc1
        self.fc2 = fc2
        self.augment = augment

    def forward(self, x):
        if self.body is not None:
            x = self.body(x)
        if self.fc1 is not None:
            x = Func.relu(self.fc1(x))
            x = Func.dropout(x, training=self.training)
        if self.fc2 is not None:
            x = self.fc2(x)
        if self.augment:
            x = torch.cat((torch.ones(x.shape[0],1).to(x.device), x), dim=1)
        return x


class CNNCifarBody(nn.Module):
    def __init__(self, outsize=120):
        super(CNNCifarBody, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, outsize)

    def forward(self, x):
        x = self.pool(Func.relu(self.conv1(x)))
        x = self.pool(Func.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = Func.relu(self.fc1(x))
        return x

# for initialization only
class CNNCifarHead():
    def __init__(self):
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 10)

class CNNCifar100Head():
    def __init__(self):
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 100)

class CNNCifar(nn.Module):
    def __init__(self, body=None, fc2=None, fc3=None, augment=False):
        super(CNNCifar, self).__init__()
        self.body = body
        self.fc2 = fc2
        self.fc3 = fc3
        self.augment = augment

    def forward(self, x):
        if self.body is not None:
            x = self.body(x)
        if self.fc2 is not None:
            x = Func.relu(self.fc2(x))
            x = Func.dropout(x, training=self.training)
        if self.fc3 is not None:
            x = self.fc3(x)
        if self.augment:
            x = torch.cat((torch.ones(x.shape[0],1).to(x.device), x), dim=1)
        return x

class CNNCifarDP(nn.Module):
    def __init__(self, b, body=None, fc2=None, fc3=None, augment=False):
        super(CNNCifarDP, self).__init__()
        self.body = body
        self.fc2 = fc2
        self.fc3 = fc3
        self.augment = augment
        self.b = abs(b)

    def forward(self, x):
        if self.body is not None:
            x = self.body(x)
        if self.fc2 is not None:
            #x = Func.relu(self.fc2(x))
            x = self.fc2(x)
            x = Func.dropout(x, training=self.training)
            x = torch.clamp(x, min=-self.b, max=self.b)
        if self.fc3 is not None:
            x = self.fc3(x)
        if self.augment:
            x = torch.cat((torch.ones(x.shape[0],1).to(x.device), x), dim=1)
        return x
