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

class ExponentialFamilyConjugatePrior():
    def __init__(self, norm_constant, chi, v=0):
        self.A = norm_constant # function of eta
        self.chi_prior = chi.clone().detach() # tensor of size m
        self.v_prior = v # constant
        self.chi_posterior = chi.clone().detach() # tensor of size m
        self.v_posterior = v # constant
        self.m = chi.shape[0]

        self.kernel = lambda eta: -((eta @ self.chi_posterior/self.v_posterior)-self.A(eta))*self.v_posterior

    def update(self, chi, v):
        self.chi_posterior += chi
        self.v_posterior += v

    def forget(self):
        self.chi_posterior = self.chi_prior.clone().detach()
        self.v_posterior = self.v_prior

    def maximize(self, tol=1e-3, max_iter=1000, lr=0.1, init=None):
        #print(self.chi_posterior)
        #print(self.v_posterior)
        if init is None:
            eta = (self.chi_posterior/self.v_posterior/2)
            eta.requires_grad = True
        else:
            eta = init
        opt = optim.Adam([eta], lr=lr)
        #opt = optim.SGD([eta],lr=lr,momentum=0.5)
        for i in range(max_iter):
            opt.zero_grad()
            last = eta.detach().clone()
            loss = self.kernel(eta)
            loss.backward()
            opt.step()

            with torch.no_grad():
                change = torch.linalg.vector_norm(last-eta,ord=np.inf)
                if change <= tol:
                    print(i, loss.item(), change.item())
                    return eta, str(change.item())
        print(max_iter, loss.item(), change.item())
        return eta, str(loss.item())

def gauss_like_constant_func(eta, num_class, m):
    res = torch.empty(num_class)
    for i in range(num_class):
        res[i] = torch.sum(eta[i*m:(i+1)*m]**2)/4
    return torch.logsumexp(res, dim=0)

def gauss_dp_std(sensitivity, epsilon, delta):
    return np.sqrt(2*np.log(1.25/delta))*sensitivity/epsilon

class FedLog(Model):
    class Parameters():
        def __init__(self, model, m, num_class, num_clients, lr, train_batch_size, test_batch_size, local_epochs, seed=None, eta_init=None, mean_err=True, b=None, dp_epsilon=None, dp_delta=None):
            self.model = model
            self.m = m
            self.num_class = num_class
            self.num_clients = num_clients
            self.lr = lr
            self.train_batch_size = train_batch_size
            self.test_batch_size = test_batch_size
            self.local_epochs = local_epochs
            self.seed = seed
            self.eta_init = eta_init
            self.mean_err = mean_err
            self.b = b
            self.dp_epsilon = dp_epsilon
            self.dp_delta = dp_delta

    def __init__(self, params):
        self.client_models = []
        for i in range(params.num_clients):
            self.client_models.append(deepcopy(params.model))
        self.m = params.m+1 if params.model.augment else params.m
        self.m_total = self.m*params.num_class
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

        self.global_eta = params.eta_init
        self.loss_func = nn.CrossEntropyLoss()
        self.eta_prob = ExponentialFamilyConjugatePrior(norm_constant=lambda eta:gauss_like_constant_func(eta, self.num_class, self.m),
                                   chi=torch.zeros(self.m_total,requires_grad=False).to(device))

        self.cur_communication = 0
        self.communication = []
        self.mean_err = params.mean_err

        self.best_vali_acc = [-1.0 for i in range(self.num_clients)]
        self.best_vali_model = [None for i in range(self.num_clients)]

        self.b = params.b
        self.dp_epsilon = params.dp_epsilon
        self.dp_delta = params.dp_delta
        self.dp = (self.dp_epsilon is not None) and (self.dp_delta is not None)
        print("is DP:", self.dp)

    def to(self, dev):
        for model in self.client_models:
            model = model.to(dev)
        if self.global_eta is not None:
            self.global_eta = self.global_eta.to(dev)
        return self

    def predict(self, model, X, log_softmax=True):
        F = model(X) @ self.global_eta # (batch_size*self.m) @ (self.m,self.num_class)
        if log_softmax:
            return Func.log_softmax(F, dim=1)
        else:
            return Func.softmax(F, dim=1)

    def local_update(self, dataset, train_index):
        print("local updating...")
        total_loss_list = []
        for i, model in enumerate(self.client_models):
            update_progress(i/self.num_clients)
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
                loss = self.loss_func(model(X) @ self.global_eta, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            total_loss /= len(train_index[i])
            #print("client", i, "training loss", total_loss)
            total_loss_list.append(str(total_loss))
        update_progress(1.0)
        return total_loss_list

    def global_update(self, dataset, train_index):
        with torch.no_grad():
            for i, model in enumerate(self.client_models):
                model.eval()
                ldr = DataLoader(Subset(dataset, train_index[i]), batch_size=self.test_batch_size, shuffle=False, sampler=None)
                ss = torch.zeros(self.m_total).to(device)
                for _, (X, y) in enumerate(ldr):
                    X = X.to(device)
                    y = y.to(device)
                    F = model(X)
                    for j in range(F.shape[0]):
                        ss[y[j]*self.m:(y[j]+1)*self.m] += F[j]
                self.eta_prob.update(chi=ss,v=len(train_index[i]))
        if self.dp:
            noise = torch.randn(self.m_total).to(device)
            noise = noise * gauss_dp_std((1+(self.m-1)*(self.b**2))**0.5, self.dp_epsilon, self.dp_delta)
            self.eta_prob.update(chi=noise, v=0)
        self.global_eta, trace = self.eta_prob.maximize(max_iter=10000, lr=0.01)
        self.global_eta = self.global_eta.view(self.num_class, self.m).T
        #print(self.global_eta.T)
        return trace

    def pretrain(self, dataset, train_index, vali_index):
        self.eta_prob.forget()
        global_trace = self.global_update(dataset, train_index)
        self.cur_communication += (2*self.m_total+1) * self.num_clients * 32
        #self.cur_communication += self.num_clients * 32
        self.communication.append(str(self.cur_communication))

    def train(self, dataset, train_index, vali_index):
        local_trace = empty_nested_lists(self.num_clients)
        for _ in range(self.local_epochs):
            loss_list = self.local_update(dataset, train_index)
            for i, (local, loss) in enumerate(zip(local_trace, loss_list)):
                local.append(loss)
        self.eta_prob.forget()
        global_trace = self.global_update(dataset, train_index)

        if vali_index is not None:
            local_correct_list = self.vali(dataset, vali_index)
            for i, acc in enumerate(local_correct_list):
                if acc > self.best_vali_acc[i]:
                    self.best_vali_acc[i] = acc
                    self.best_vali_model[i] = deepcopy(self.client_models[i])
            print("cur vali acc:", np.mean(local_correct_list), "best vali acc:", np.mean(self.best_vali_acc))

        #plt.plot(list(range(len(trace))), trace)
        self.cur_communication += 2*self.m_total*self.num_clients*32
        self.communication.append(str(self.cur_communication))
        return global_trace, local_trace

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
        print("testing ...")
        for i, model in enumerate(active_models):
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
            #return str(np.mean(local_correct_list)), str(np.mean(global_correct_list))
            return str(np.mean(local_correct_list))
        else:
            return local_correct_list, global_correct_list
