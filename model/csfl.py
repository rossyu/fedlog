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

def flatten_params(parameters):
    """
    flattens all parameters into a single column vector. Returns the dictionary to recover them
    :param: parameters: a generator or list of all the parameters
    :return: a dictionary: {"params": [#params, 1],
    "indices": [(start index, end index) for each param] **Note end index in uninclusive**

    """
    l = [torch.flatten(p) for p in parameters]
    indices = []
    s = 0
    for p in l:
        size = p.shape[0]
        indices.append((s, s+size))
        s += size
    flat = torch.cat(l).view(-1, 1)
    return flat, indices

def recover_flattened(flat_params, indices, model):
    """
    Gives a list of recovered parameters from their flattened form
    :param flat_params: [#params, 1]
    :param indices: a list detaling the start and end index of each param [(start, end) for param]
    :param model: the model that gives the params with correct shapes
    :return: the params, reshaped to the ones in the model, with the same order as those in the model
    """
    l = [flat_params[s:e] for (s, e) in indices]
    for i, p in enumerate(model.parameters()):
        p.data.copy_(l[i].view(*p.shape))
    #return l

@torch.no_grad()
def IHT(x, phi, sparcity):
    y = torch.zeros(phi.shape[1],1).to(device)
    y_last = deepcopy(y)
    diff = 1.0
    trace = []
    count = 0
    while diff > 1e-4 and count < 1000:
        count += 1
        y_last = deepcopy(y)
        y = spar(y+phi.T@(x-phi@y), sparcity, False)
        diff = (y-y_last).norm().item()
    print(count)
    return y

@torch.no_grad()
def BIHT(x, phi, sparcity):
    y = torch.zeros(phi.shape[1],1).to(device)
    y_last = deepcopy(y)
    count = 0
    diff = 1.0
    while diff > 1e-4 and count < 1000:
        count += 1
        y_last = deepcopy(y)
        y = spar(y+phi.T@(x-torch.sign(phi@y)), sparcity, False)
        diff = (y-y_last).norm()
    #print(count, diff)
    return y

@torch.no_grad()
def spar(dense, sparcity, is_absolute=True):
    x = deepcopy(dense)
    if is_absolute:
        x[torch.abs(x)<sparcity] = 0
    else:
        threshold = torch.kthvalue(torch.abs(x), sparcity,0)[0].item()
        x[torch.abs(x)<threshold] = 0
    return x

class CSFL(Model):
    class Parameters():
        def __init__(self, global_model, num_class, num_clients, lr, train_batch_size, test_batch_size, local_epochs, sparcity, reduced_dim, lr_multiplier, binary, seed=None, mean_err=True):
            self.global_model = global_model
            self.num_class = num_class
            self.num_clients = num_clients
            self.lr = lr
            self.train_batch_size = train_batch_size
            self.test_batch_size = test_batch_size
            self.local_epochs = local_epochs
            self.sparcity = sparcity
            self.reduced_dim = reduced_dim
            self.lr_multiplier = lr_multiplier
            self.binary = binary
            self.seed = seed
            self.mean_err = mean_err

    def __init__(self, params):
        self.client_models = []
        for i in range(params.num_clients):
            self.client_models.append(deepcopy(params.global_model))
        self.wt, self.indices = flatten_params(self.client_models[0].parameters())

        self.m = sum(p.numel() for p in params.global_model.parameters() if p.requires_grad)
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

        if params.reduced_dim < 1:
            self.reduced_dim = int(self.m * params.reduced_dim)
        else:
            self.reduced_dim = params.reduced_dim
        self.sparcity = params.sparcity
        self.lr_multiplier = params.lr_multiplier
        self.binary = params.binary

    def to(self, dev):
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
            ldr = DataLoader(Subset(dataset, train_index[i]), batch_size=self.train_batch_size,
                          sampler=RandomSampler(Subset(dataset, train_index[i]), generator=self.sampler))
            total_loss = 0.0
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            model.train()
            for _, (X, y) in enumerate(ldr):
                X = X.to(device)
                y = y.to(device)
                model.zero_grad()
                loss = self.loss_func(model(X), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            total_loss /= len(train_index[i])
            #print("client", i, "training loss", total_loss)
            total_loss_list.append(str(total_loss))
        return total_loss_list

    @torch.no_grad()
    def global_update_phase1(self, dataset, train_index, At):
        yt = torch.zeros(self.reduced_dim,1).to(device)
        et = []
        mean_sparcity = 0
        for i, model in enumerate(self.client_models):
            wti, _ = flatten_params(model.parameters())
            hti = wti - self.wt
            sti = spar(hti, self.sparcity)
            mean_sparcity += torch.count_nonzero(sti).item()
            eti = hti - sti
            et.append(eti)
            if self.binary:
                yti = torch.sign(At @ sti)
                self.cur_communication += 2*self.reduced_dim
            else:
                yti = At @ sti
                self.cur_communication += 2*self.reduced_dim*32
            yt += yti / self.num_clients
        mean_sparcity = int(mean_sparcity / self.num_clients)
        if self.binary:
            st_tilde = BIHT(yt, At, mean_sparcity)
        else:
            st_tilde = IHT(yt, At, mean_sparcity)
        self.wt = self.wt + self.lr * st_tilde * self.lr_multiplier
        for model in self.client_models:
            recover_flattened(self.wt, self.indices, model)
        return et

    @torch.no_grad()
    def global_update_phase2(self, dataset, train_index, et):
        rt = torch.zeros_like(et[0]).to(device)
        for i, model in enumerate(self.client_models):
            wti_tilde, _ = flatten_params(model.parameters())
            hti_tilde = wti_tilde - self.wt
            rti_tilde = torch.sign(et[i]+hti_tilde)
            rt += rti_tilde / self.num_clients
        self.cur_communication += 2*self.reduced_dim*self.num_clients
        rt = torch.sign(rt)
        self.wt = self.wt + self.lr * rt * self.lr_multiplier
        #self.wt = self.wt + self.lr * rt
        for model in self.client_models:
            recover_flattened(self.wt, self.indices, model)

    def train(self, dataset, train_index, vali_index):
        local_trace = empty_nested_lists(self.num_clients)
        for _ in range(self.local_epochs):
            loss_list = self.local_update(dataset, train_index)
            #for i, (local, loss) in enumerate(zip(local_trace, loss_list)):
                #local.append(loss)
        At = torch.randn(self.reduced_dim, self.m).to(device)
        S = torch.linalg.svdvals(At)
        At /= S[0]
        self.cur_communication += self.num_clients*32
        et = self.global_update_phase1(dataset, train_index, At)
        for _ in range(self.local_epochs):
            loss_list = self.local_update(dataset, train_index)
            #for i, (local, loss) in enumerate(zip(local_trace, loss_list)):
                #local.append(loss)
        self.global_update_phase2(dataset, train_index, et)
        self.communication.append(str(self.cur_communication))
        #plt.plot(list(range(len(trace))), trace)
        return local_trace

    def pretrain(self, dataset, train_index, vali_index):
        self.cur_communication += self.num_clients * 32
        self.communication.append(str(self.cur_communication))
        return None

    def test(self, dataset, test_index):
        with torch.no_grad():
            local_correct_list = []
            global_correct_list = []
            print("testing...")
            for i, model in enumerate(self.client_models):
                update_progress(i/self.num_clients)
                model.eval()
                correct = 0

                ldr = DataLoader(Subset(dataset, test_index[i]), batch_size=self.test_batch_size, shuffle=False, sampler=None)
                for j, (X, y) in enumerate(ldr):
                    X = X.to(device)
                    y = y.to(device)
                    y_pred = self.predict(model, X).max(dim=1)[1]
                    correct += y_pred.eq(y).int().sum().to("cpu")
                local_correct_list.append(100.0*correct/len(test_index[i]))
            update_progress(1.0)
        if self.mean_err:
            return str(np.mean(local_correct_list))
        else:
            return local_correct_list, global_correct_list

