import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as Func
from torch.utils.data import DataLoader, Subset, RandomSampler
from copy import deepcopy
import numpy as np

from model.base import Model
from utils import update_progress, empty_nested_lists

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FedRep(Model):
    class Parameters():
        def __init__(self, local_model, global_model, num_class, num_clients, lr, train_batch_size, test_batch_size, local_epochs, local_head_epochs, seed=None, mean_err=True):
            self.local_model = local_model
            self.global_model = global_model
            self.num_class = num_class
            self.num_clients = num_clients
            self.lr = lr
            self.train_batch_size = train_batch_size
            self.test_batch_size = test_batch_size
            self.local_epochs = local_epochs
            self.local_head_epochs = local_head_epochs
            self.seed = seed
            self.mean_err = mean_err

    def __init__(self, params):
        self.global_model = deepcopy(params.global_model)
        self.client_models = []
        for i in range(params.num_clients):
            self.client_models.append([deepcopy(params.global_model), deepcopy(params.local_model)])

        self.m = sum(p.numel() for p in params.global_model.parameters() if p.requires_grad)
        self.num_class = params.num_class
        self.num_clients = params.num_clients
        self.lr = params.lr
        self.train_batch_size = params.train_batch_size
        self.test_batch_size = params.test_batch_size
        self.local_epochs = params.local_epochs
        self.local_head_epochs = params.local_head_epochs
        self.seed = params.seed
        if self.seed is not None:
            self.sampler = torch.Generator().manual_seed(self.seed)
        else:
            self.sampler = None
        self.mean_err = params.mean_err
        self.loss_func = nn.CrossEntropyLoss()
        self.cur_communication = 0
        self.communication = []

    def to(self, dev):
        self.global_model = self.global_model.to(dev)
        for model in self.client_models:
            model[0] = model[0].to(dev)
            model[1] = model[1].to(dev)
        return self

    def predict(self, model, X, log_softmax=True):
        F = model[1](model[0](X)) # (batch_size*self.num_class)
        if log_softmax:
            return Func.log_softmax(F, dim=1)
        else:
            return Func.softmax(F, dim=1)

    def local_head_update(self, ldr, model):
        model[0].requires_grad_(False)
        model[0].eval()
        model[1].requires_grad_(True)
        model[1].train()
        optimizer = torch.optim.Adam(model[1].parameters(), lr=self.lr*5)
        for _, (X, y) in enumerate(ldr):
            X = X.to(device)
            y = y.to(device)
            # Head training
            model[1].zero_grad()
            #log_probs = self.predict(model, X)
            loss = self.loss_func(model[1](model[0](X)), y)
            loss.backward()
            optimizer.step()

    def local_update(self, dataset, train_index):
        total_loss_list = []
        for i, model in enumerate(self.client_models):
            ldr = DataLoader(Subset(dataset, train_index[i]), batch_size=self.train_batch_size,
                          sampler=RandomSampler(Subset(dataset, train_index[i]), generator=self.sampler))
            total_loss = 0.0

            optimizer = torch.optim.Adam(model[0].parameters(), lr=self.lr)
            for _ in range(self.local_head_epochs):
                self.local_head_update(ldr, model)

            model[1].requires_grad_(False)
            model[1].eval()
            model[0].requires_grad_(True)
            model[0].train()
            for _, (X, y) in enumerate(ldr):
                X = X.to(device)
                y = y.to(device)
                # Body training
                model[0].zero_grad()
                loss = self.loss_func(model[1](model[0](X)), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            total_loss /= len(train_index[i])
            #print("client", i, "training loss", total_loss)
            total_loss_list.append(str(total_loss))
        return total_loss_list

    def global_update(self, dataset, train_index):
        with torch.no_grad():
            global_paras = dict(self.global_model.named_parameters())
            for key in global_paras:
                average = torch.zeros_like(global_paras[key].data)
                for i, model in enumerate(self.client_models):
                    local_paras = dict(model[0].named_parameters())[key].data
                    average += local_paras / self.num_clients
                global_paras[key].data.copy_(average)
            for model in self.client_models:
                model[0] = deepcopy(self.global_model)
                model[0].requires_grad_(True)
                model[1].requires_grad_(True)

    def pretrain(self, dataset, train_index, vali_index):
        self.cur_communication += self.num_clients*32
        self.communication.append(str(self.cur_communication))
        return None

    def train(self, dataset, train_index, vali_index):
        self.global_update(dataset, train_index)
        local_trace = empty_nested_lists(self.num_clients)
        for _ in range(self.local_epochs):
            loss_list = self.local_update(dataset, train_index)
            for i, (local, loss) in enumerate(zip(local_trace, loss_list)):
                local.append(loss)
        #plt.plot(list(range(len(trace))), trace)
        self.cur_communication += 2*self.m*self.num_clients*32
        self.communication.append(str(self.cur_communication))
        return local_trace

    def test(self, dataset, test_index):
        with torch.no_grad():
            local_correct_list = []
            global_correct_list = []
            print("testing...")
            for i, model in enumerate(self.client_models):
                update_progress(i/self.num_clients)
                model[0].eval()
                model[1].eval()
                correct = 0

                ldr = DataLoader(Subset(dataset, test_index[i]), batch_size=self.test_batch_size, shuffle=False, sampler=None)
                for j, (X, y) in enumerate(ldr):
                    X = X.to(device)
                    y = y.to(device)
                    y_pred = self.predict(model, X).max(dim=1)[1]
                    correct += y_pred.eq(y).int().sum()
                local_correct_list.append(100.0*correct/len(test_index[i]))
            update_progress(1.0)
        if self.mean_err:
            return str(np.mean(local_correct_list))
        else:
            return local_correct_list, global_correct_list
