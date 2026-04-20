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

class FedAvg(Model):
    class Parameters():
        def __init__(self, model, num_class, num_clients, lr, train_batch_size, test_batch_size, local_epochs, seed=None, mean_err=True):
            self.model = model
            self.num_class = num_class
            self.num_clients = num_clients
            self.lr = lr
            self.train_batch_size = train_batch_size
            self.test_batch_size = test_batch_size
            self.local_epochs = local_epochs
            self.seed = seed
            self.mean_err = mean_err

    def __init__(self, params):
        self.global_model = deepcopy(params.model)
        self.client_models = []
        for i in range(params.num_clients):
            self.client_models.append(deepcopy(params.model))
        self.m = sum(p.numel() for p in params.model.parameters() if p.requires_grad)
        self.num_class = params.num_class
        self.num_clients = params.num_clients
        self.lr = params.lr
        self.train_batch_size = params.train_batch_size
        self.test_batch_size = params.test_batch_size
        self.local_epochs = params.local_epochs
        self.seed = params.seed
        if self.seed is not None:
            self.sampler = torch.Generator().manual_seed(self.seed)
        else:
            self.sampler = None
        self.mean_err = params.mean_err
        self.loss_func = nn.CrossEntropyLoss()
        self.cur_communication = 0 
        self.communication = []

        self.best_vali_acc = [-1.0 for i in range(self.num_clients)]
        self.best_vali_model = [None for i in range(self.num_clients)]

    def to(self, dev):
        self.global_model = self.global_model.to(dev)
        for model in self.client_models:
            model = model.to(dev)
        return self

    def predict(self, model, X, log_softmax=True):
        F = model(X) # (batch_size*self.num_class)
        if log_softmax:
            return Func.log_softmax(F, dim=1)
        else:
            return Func.softmax(F, dim=1)

    def local_update(self, dataset, train_index):
        total_loss_list = []
        for i, model in enumerate(self.client_models):
            model.train()
            ldr = DataLoader(Subset(dataset, train_index[i]), batch_size=self.train_batch_size,
                      sampler=RandomSampler(Subset(dataset, train_index[i]), generator=self.sampler))
            total_loss = 0.0

            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            for _, (X, y) in enumerate(ldr):
                X = X.to(device)
                y = y.to(device)
                model.zero_grad()
                #log_probs = self.predict(model, X)
                #loss = self.loss_func(log_probs, y)
                loss = self.loss_func(model(X), y)
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
                    local_paras = dict(model.named_parameters())[key].data
                    average += local_paras / self.num_clients 
                global_paras[key].data.copy_(average)
            for i in range(self.num_clients):
                self.client_models[i] = deepcopy(self.global_model)

    def pretrain(self, dataset, train_index, vali_index):
        self.cur_communication += self.num_clients*32
        self.communication.append(str(self.cur_communication))

    def train(self, dataset, train_index, vali_index):
        local_trace = empty_nested_lists(self.num_clients)
        for _ in range(self.local_epochs):
            loss_list = self.local_update(dataset, train_index)
            for i, (local, loss) in enumerate(zip(local_trace, loss_list)):
                local.append(loss)
        self.global_update(dataset, train_index)

        if vali_index is not None:
            local_correct_list = self.vali(dataset, vali_index)
            for i, acc in enumerate(local_correct_list):
                if acc > self.best_vali_acc[i]:
                    self.best_vali_acc[i] = acc
                    self.best_vali_model[i] = deepcopy(self.client_models[i])
            print("cur vali acc:", np.mean(local_correct_list), "best vali acc:", np.mean(self.best_vali_acc))

        #plt.plot(list(range(len(trace))), trace)
        self.cur_communication += 2*self.m*self.num_clients*32
        self.communication.append(str(self.cur_communication))
        return local_trace

    @torch.no_grad()
    def vali(self, dataset, vali_index):
        local_correct_list = []
        print("validation ...")
        for i, model in enumerate(self.client_models):
            update_progress(i/self.num_clients)
            model.eval()
            correct = 0
            ldr = DataLoader(Subset(dataset, vali_index[i]), batch_size=self.test_batch_size, shuffle=False, sampler=None)
            for j, (X, y) in enumerate(ldr):
                X = X.to(device)
                y = y.to(device)
                y_pred = self.predict(model, X).max(dim=1)[1]
                correct += y_pred.eq(y).int().sum().to("cpu")
            local_correct_list.append(100.0*correct/len(vali_index[i]))
        update_progress(1.0)
        return local_correct_list

    @torch.no_grad()
    def test(self, dataset, test_index):
        active_models = self.client_models if self.best_vali_model[0] is None else self.best_vali_model
        local_correct_list = []
        global_correct_list = []
        print("testing...")
        for i, model in enumerate(active_models):
            update_progress(i/self.num_clients)
            model.eval()
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
            #return str(np.mean(local_correct_list)), str(np.mean(global_correct_list))
            return str(np.mean(local_correct_list))
        else:
            return local_correct_list, global_correct_list
