import json
import torch
from pprint import pprint
from utils import val_to_str

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Experiment():
    class Parameters():
        def __init__(self):
            self.show_metric_train = True
            self.global_epochs = 10
            self.pretrain = True
            self.path = "./"

    def __init__(self, params):
        self.show_metric_train = params.show_metric_train
        self.global_epochs = params.global_epochs
        self.pretrain = params.pretrain
        self.path = params.path

        self.pretrain_trace = {}
        self.trace = {}
        self.train_err = {}
        self.test_err = {}
        self.communication = {}

        self.data_list = []
        self.data_name_list = []
        self.data_params_list = []
        self.n_datasets = 0

        self.model_list = []
        self.model_name_list = []
        self.model_params_list = []
        self.n_models = 0

    def register_dataset(self, name, data_class, data_params):
        self.data_list.append(data_class)
        self.data_name_list.append(name)
        self.data_params_list.append(data_params)
        self.n_datasets += 1

    def register_model(self, name, model_class, model_params):
        self.model_list.append(model_class)
        self.model_name_list.append(name)
        self.model_params_list.append(model_params)
        self.n_models += 1

    def save_results(self, dname):
        #with open(self.path+dname+"_pretrain_trace.json", 'w+') as fp:
            #json.dump(self.pretrain_trace, fp, indent=4)
        #with open(self.path+dname+"_trace.json", 'w+') as fp:
            #json.dump(self.trace, fp, indent=4)
        #with open(self.path+dname+"_train_err.json", 'w+') as fp:
            #json.dump(self.train_err, fp, indent=4)
        with open(self.path+dname+"_test_err.json", 'w+') as fp:
            json.dump(self.test_err, fp, indent=4)
        with open(self.path+dname+"_communication_cost.json","w+") as fp:
            json.dump(self.communication, fp, indent=4)
        with open(self.path+dname+"_setting.json","a+") as fp:
            fp.write("total global epochs: "+str(self.global_epochs))
            fp.write("\n")
            fp.write("model settings:\n")
            params = val_to_str(dict(vars(self.model_params_list[0])))
            json.dump(params, fp, indent=4)
            fp.write("\n")
            fp.write("data settings:\n")
            params = val_to_str(dict(vars(self.data_params_list[0])))
            json.dump(params, fp, indent=4)

    def run(self):
        params = val_to_str(dict(vars(self.model_params_list[0])))
        print(params)
        params = val_to_str(dict(vars(self.data_params_list[0])))
        print(params)
        for di in range(self.n_datasets):
            dname = self.data_name_list[di]
            dparams = self.data_params_list[di]
            data = self.data_list[di](dparams)
            print()
            print("######################################################")
            print("Running experiments with dataset", dname)
            print("######################################################")
            train_index, test_index, vali_index = data.read_and_split()
            print("Data successfully loaded and split")

            for moi in range(self.n_models):
                mname = self.model_name_list[moi]
                mparams = self.model_params_list[moi]
                model = self.model_list[moi](mparams).to(device)
                print()
                print("-----------------------------------------------------------")
                print("Current model:", mname)
                print("-----------------------------------------------------------")
                self.trace[mname] = []
                self.train_err[mname] = []
                self.test_err[mname] = []
                self.communication[mname] = []

                if self.pretrain:
                    print("pretraining...")
                    self.pretrain_trace[mname] = model.pretrain(data.trainset, train_index, vali_index)
                    if self.show_metric_train:
                        self.train_err[mname].append(model.test(data.trainset, train_index))
                    if test_index is not None:
                        self.test_err[mname].append(model.test(data.testset, test_index))
                        print("test acc:", self.test_err[mname][-1])

                for i in range(self.global_epochs):
                    print("epoch", i)
                    model.train(data.trainset, train_index, vali_index)
                    #self.trace[mname].append(model.train(data.trainset, train_index, vali_index))
                    if self.show_metric_train:
                        self.train_err[mname].append(model.test(data.trainset, train_index))
                    if test_index is not None:
                        self.test_err[mname].append(model.test(data.testset, test_index))
                    if i % 1 == 0:
                        if self.show_metric_train:
                            print("train acc:", self.train_err[mname][-1])
                        print("test acc:", self.test_err[mname][-1])
                    self.communication[mname]=model.communication
                    self.save_results(dname)
                if self.show_metric_train:
                    print("train acc:", self.train_err[mname])
                print("test acc:", self.test_err[mname])
                print("total communication cost:", model.communication)
                self.communication[mname]=model.communication
                self.save_results(dname)
                print("")

            print()
            for key in self.communication:
                print(key)
                if self.show_metric_train:
                    print("train acc:", self.train_err[key])
                print("test acc:", self.test_err[key])
                print("communication cost:", self.communication[key])
