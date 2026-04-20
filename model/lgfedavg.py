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

class LGFedAvg(Model):
    class Parameters():
        def __init__(self, local_model, global_model, num_class, num_clients, lr, train_batch_size, test_batch_size, local_epochs, avg_epochs=0, seed=None, mean_err=True):
            self.local_model = local_model
            self.global_model = global_model
            self.num_class = num_class
            self.num_clients = num_clients
            self.lr = lr
            self.train_batch_size = train_batch_size
            self.test_batch_size = test_batch_size
            self.local_epochs = local_epochs
            self.avg_epochs = avg_epochs
            self.seed = seed
            self.mean_err = mean_err

    def __init__(self, params):
        self.global_model = deepcopy(params.global_model)
        self.client_models = []
        for i in range(params.num_clients):
            self.client_models.append([deepcopy(params.local_model), deepcopy(params.global_model)])

        self.m = sum(p.numel() for p in params.global_model.parameters() if p.requires_grad)
        self.m_whole = self.m + sum(p.numel() for p in params.local_model.parameters() if p.requires_grad)
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
        self.communication = []
        self.cur_communication = 0
        self.avg_epochs = params.avg_epochs
        self.remain_avg_epochs = params.avg_epochs

        self.best_vali_acc = [-1.0 for i in range(self.num_clients)]
        self.best_vali_model = [None for i in range(self.num_clients)]
        #self.best_pre_acc = -1.0
        #self.best_pre_model = None
        #self.global_avg_model = [deepcopy(self.client_models[0][0]), deepcopy(self.client_models[0][1])]

    def to(self, dev):
        self.global_model = self.global_model.to(dev)
        for model in self.client_models:
            model[0] = model[0].to(dev)
            model[1] = model[1].to(dev)
        #self.global_avg_model[0] = self.global_avg_model[0].to(dev)
        #self.global_avg_model[1] = self.global_avg_model[1].to(dev)
        return self

    def predict(self, model, X, log_softmax=True):
        F = model[1](model[0](X)) # (batch_size*self.num_class)
        if log_softmax:
            return Func.log_softmax(F, dim=1)
        else:
            return Func.softmax(F, dim=1)

    def local_update(self, dataset, train_index, chosen_clients):
        if self.remain_avg_epochs > 0:
            return self.local_update_avg(dataset, train_index, chosen_clients)
        else:
            #return self.local_update_lg(dataset, train_index)
            return self.local_update_avg(dataset, train_index, chosen_clients)

    def global_update(self, dataset, train_index, chosen_clients):
        if self.remain_avg_epochs > 0:
            return self.global_update_avg(dataset, train_index, chosen_clients)
        else:
            return self.global_update_lg(dataset, train_index, chosen_clients)

    def global_update_lg(self, dataset, train_index, chosen_clients):
        with torch.no_grad():
            total_size = 0
            for i in chosen_clients:
                total_size += len(train_index[i])
            global_paras = dict(self.global_model.named_parameters())
            for key in global_paras:
                average = torch.zeros_like(global_paras[key].data)
                for i in chosen_clients:
                    local_paras = dict(self.client_models[i][1].named_parameters())[key].data
                    average += local_paras * (float(len(train_index[i])) / total_size)
                global_paras[key].data.copy_(average)
            for model in self.client_models:
                model[1] = deepcopy(self.global_model)
                #model[1].requires_grad_(True)

    def local_update_avg(self, dataset, train_index, chosen_clients):
        total_loss_list = []
        #for i, model in enumerate(self.client_models):
        for i in chosen_clients:
            #self.client_models[i] = [deepcopy(self.global_avg_model[0]), deepcopy(self.global_avg_model[1])]
            model = self.client_models[i]
            model[0].train()
            model[1].train()
            ldr = DataLoader(Subset(dataset, train_index[i]), batch_size=self.train_batch_size,
                          sampler=RandomSampler(Subset(dataset, train_index[i]), generator=self.sampler))
            total_loss = 0.0

            params = list(model[0].parameters()) + list(model[1].parameters())
            optimizer = torch.optim.Adam(params, lr=self.lr)
            #optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.5)
            for _, (X, y) in enumerate(ldr):
                X = X.to(device)
                y = y.to(device)
                model[0].zero_grad()
                model[1].zero_grad()
                loss = self.loss_func(model[1](model[0](X)), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            total_loss /= len(train_index[i])
            #print("client", i, "training loss", total_loss)
            total_loss_list.append(str(total_loss))
        return total_loss_list

    def global_update_avg(self, dataset, train_index, chosen_clients):
        with torch.no_grad():
            total_size = 0
            for i in chosen_clients:
                total_size += len(train_index[i])
            global_paras = dict(self.global_avg_model[1].named_parameters())
            for key in global_paras:
                average = torch.zeros_like(global_paras[key].data)
                for i in chosen_clients:
                    model = self.client_models[i]
                    local_paras = dict(model[1].named_parameters())[key].data
                    average += local_paras * (float(len(train_index[i])) / total_size)
                global_paras[key].data.copy_(average)

            global_paras = dict(self.global_avg_model[0].named_parameters())
            for key in global_paras:
                average = torch.zeros_like(global_paras[key].data)
                for i in chosen_clients:
                    model = self.client_models[i]
                    local_paras = dict(model[0].named_parameters())[key].data
                    average += local_paras * (float(len(train_index[i])) / total_size)
                global_paras[key].data.copy_(average)

    def pretrain(self, dataset, train_index, vali_index):
        self.cur_communication += self.num_clients*32
        self.communication.append(str(self.cur_communication))
        return None

    def train(self, dataset, train_index, vali_index):
        train_index_copy = deepcopy(train_index)
        local_trace = empty_nested_lists(self.num_clients)
        #chosen_clients = np.random.choice(range(self.num_clients), int(self.num_clients/10), replace=False)
        chosen_clients = np.arange(self.num_clients)
        if self.remain_avg_epochs > 0:
            for i in chosen_clients:
                train_index_copy[i] = np.concatenate((train_index_copy[i], vali_index[i]))
        for _ in range(self.local_epochs):
            loss_list = self.local_update(dataset, train_index_copy, chosen_clients)
            for i, (local, loss) in enumerate(zip(local_trace, loss_list)):
                local.append(loss)
        self.global_update(dataset, train_index_copy, chosen_clients)
        if self.remain_avg_epochs <= 0 and vali_index is not None:
            local_correct_list = self.vali(dataset, vali_index)
            for i, acc in enumerate(local_correct_list):
                if acc >= self.best_vali_acc[i]:
                    self.best_vali_acc[i] = acc
                    self.best_vali_model[i] = deepcopy(self.client_models[i])
            print("cur vali acc:", np.mean(local_correct_list), "best vali acc:", np.mean(self.best_vali_acc))
        #if self.remain_avg_epochs == 1:
            #self.client_models = []
            #for i in range(self.num_clients):
                #self.client_models.append(self.best_pre_model)
        if self.remain_avg_epochs > 0:
            self.cur_communication += 2*self.m_whole*self.num_clients*32
            self.communication.append(str(self.cur_communication))
            self.remain_avg_epochs -= 1
        else:
            self.cur_communication += 2*self.m*self.num_clients*32
            self.communication.append(str(self.cur_communication))
        #plt.plot(list(range(len(trace))), trace)
        return local_trace

    @torch.no_grad()
    def vali(self, dataset, vali_index):
        local_correct_list = []
        print("validation ...")
        for i, model in enumerate(self.client_models):
            update_progress(i/self.num_clients)
            model[0].eval()
            model[1].eval()
            correct = 0
            ldr = DataLoader(Subset(dataset, vali_index[i]), batch_size=self.test_batch_size, shuffle=False, sampler=None)
            for j, (X, y) in enumerate(ldr):
                X = X.to(device)
                y = y.to(device)
                y_pred = self.predict(model, X).max(dim=1)[1]
                correct += y_pred.eq(y).int().sum().item()
            local_correct_list.append(100.0*correct/len(vali_index[i]))
        update_progress(1.0)
        return local_correct_list

    @torch.no_grad()
    def test(self, dataset, test_index):
        active_models = self.client_models if self.best_vali_model[0] is None else self.best_vali_model
        local_correct_list = []
        global_correct_list = []
        print("testing...")
        #if self.best_vali_model[0] is not None:
        if True:
            total_size = len(dataset)
            for i, model in enumerate(active_models):
                update_progress(i/self.num_clients)
                model[0].eval()
                model[1].eval()
                correct = 0
                ldr = DataLoader(Subset(dataset, test_index[i]), batch_size=self.test_batch_size, shuffle=False, sampler=None)
                for j, (X, y) in enumerate(ldr):
                    X = X.to(device)
                    y = y.to(device)
                    y_pred = self.predict(model, X).max(dim=1)[1]
                    correct += y_pred.eq(y).int().sum().item()
                local_correct_list.append(100.0*correct/len(test_index[i]))
            update_progress(1.0)
        else:
            self.global_avg_model[0].eval()
            self.global_avg_model[1].eval()
            for i in range(self.num_clients):
                correct = 0
                ldr = DataLoader(Subset(dataset, test_index[i]), batch_size=self.test_batch_size, shuffle=False, sampler=None)
                for j, (X, y) in enumerate(ldr):
                    X = X.to(device)
                    y = y.to(device)
                    y_pred = self.predict(self.global_avg_model, X).max(dim=1)[1]
                    correct += y_pred.eq(y).int().sum().item()
                local_correct_list.append(100.0*correct/len(test_index[i]))
        mean_acc = np.mean(local_correct_list)
        if self.remain_avg_epochs > 0 and mean_acc > self.best_pre_acc:
            self.best_pre_acc = mean_acc
            self.best_pre_model = deepcopy(active_models[0])
        if self.mean_err:
            #return str(np.mean(local_correct_list)), str(np.mean(global_correct_list))
            return str(mean_acc)
        else:
            return local_correct_list, global_correct_list
