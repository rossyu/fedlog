import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from data.base import Data, noniid, iid

class FedMNIST(Data):
    CLASS_SIZE = 10
    class Parameters():
        def __init__(self, path="~/fedlog/data/mnist", test_size=None, validation_size=None,
                     normalize=True, seed=None, num_clients=100, class_per_client=2, subset=1.0):
            self.path = path
            self.test_size = test_size
            self.validation_size = validation_size
            self.normalize = normalize
            self.seed = seed
            self.num_clients = num_clients
            self.class_per_client = class_per_client
            self.subset = subset

    def __init__(self, params):
        super(FedMNIST, self).__init__()
        self.params = params

    def read_and_split(self):
        if self.params.normalize:
            transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
        else:
            transform = transforms.ToTensor()
        self.trainset = torchvision.datasets.MNIST(root=self.params.path, train=True,
                                download=True, transform=transform)
        self.testset = torchvision.datasets.MNIST(root=self.params.path, train=False,
                                download=True, transform=transform)
        if self.params.class_per_client == self.CLASS_SIZE:
            dict_train = iid(self.trainset, self.params.num_clients, seed=self.params.seed)
            dict_test = iid(self.testset, self.params.num_clients, seed=self.params.seed)
        else:
            dict_train, rand_set_all = noniid(self.trainset, self.params.num_clients,
                                self.params.class_per_client, seed=self.params.seed, subset=60000*self.params.subset)
            dict_test, rand_set_all = noniid(self.testset, self.params.num_clients,
                                self.params.class_per_client, rand_set_all, seed=self.params.seed, subset=10000)
        return dict_train[0], dict_test[0], dict_train[1]

class FedCIFAR10():
    CLASS_SIZE = 10
    class Parameters():
        def __init__(self, path="~/fedlog/data/cifar10", test_size=None, validation_size=None,
                     normalize=True, seed=None, num_clients=100, class_per_client=2):
            self.path = path
            self.test_size = test_size
            self.validation_size = validation_size
            self.normalize = normalize
            self.seed = seed
            self.num_clients = num_clients
            self.class_per_client = class_per_client

    def __init__(self, params):
        super(FedCIFAR10, self).__init__()
        self.params = params

    def read_and_split(self):
        if self.params.normalize:
            train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])])
            transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                        std=[0.267, 0.256, 0.276])])
        else:
            transform = transforms.ToTensor()
        self.trainset = torchvision.datasets.CIFAR10(root=self.params.path, train=True,
                                download=True, transform=train_transform)
        self.testset = torchvision.datasets.CIFAR10(root=self.params.path, train=False,
                                download=True, transform=transform)
        self.trainset.targets = torch.tensor(self.trainset.targets)
        self.testset.targets = torch.tensor(self.testset.targets)

        if self.params.class_per_client == self.CLASS_SIZE:
            dict_train = iid(self.trainset, self.params.num_clients, seed=self.params.seed)
            dict_test = iid(self.testset, self.params.num_clients, seed=self.params.seed)
        else:
            dict_train, rand_set_all = noniid(self.trainset, self.params.num_clients,
                                self.params.class_per_client, seed=self.params.seed, subset=np.inf)
            dict_test, rand_set_all = noniid(self.testset, self.params.num_clients,
                                self.params.class_per_client, rand_set_all, seed=self.params.seed, subset=np.inf)
        return dict_train[0], dict_test[0], dict_train[1]

class FedCIFAR100():
    CLASS_SIZE = 100
    class Parameters():
        def __init__(self, path="~/fedlog/data/cifar100", test_size=None, validation_size=None,
                     normalize=True, seed=None, num_clients=100, class_per_client=10):
            self.path = path
            self.test_size = test_size
            self.validation_size = validation_size
            self.normalize = normalize
            self.seed = seed
            self.num_clients = num_clients
            self.class_per_client = class_per_client

    def __init__(self, params):
        super(FedCIFAR100, self).__init__()
        self.params = params

    def read_and_split(self):
        if self.params.normalize:
            train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                           std=[0.2023, 0.1994, 0.2010])])
            transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2023, 0.1994, 0.2010])])
        else:
            transform = transforms.ToTensor()
        self.trainset = torchvision.datasets.CIFAR100(root=self.params.path, train=True,
                                download=True, transform=train_transform)
        self.testset = torchvision.datasets.CIFAR100(root=self.params.path, train=False,
                                download=True, transform=transform)
        self.trainset.targets = torch.tensor(self.trainset.targets)
        self.testset.targets = torch.tensor(self.testset.targets)

        if self.params.class_per_client == self.CLASS_SIZE:
            dict_train = iid(self.trainset, self.params.num_clients, seed=self.params.seed)
            dict_test = iid(self.testset, self.params.num_clients, seed=self.params.seed)
        else:
            dict_train, rand_set_all = noniid(self.trainset, self.params.num_clients,
                                self.params.class_per_client, seed=self.params.seed, subset=np.inf)
            dict_test, rand_set_all = noniid(self.testset, self.params.num_clients,
                                self.params.class_per_client, rand_set_all, seed=self.params.seed, subset=np.inf)
        return dict_train[0], dict_test[0], dict_train[1]
