import sys

from utils import setup_seed
from model.fedlog import FedLog
from model.base import * 
from data.regular import *
from expbase import Experiment
import data.base

SEED = int(sys.argv[1])
DATASET = sys.argv[2]

setup_seed(SEED)

if DATASET == "mnist":
    data.base.VALI_PROPORTION = 0
    NUM_CLIENTS=50
    params = Experiment.Parameters()
    params.show_metric_train = False
    params.global_epochs = 100
    params.pretrain = True
    params.path = "./res/communication_cost/mnist/fedlog_"+str(SEED)+"_"
    exp = Experiment(params)

    exp.register_dataset(DATASET, FedMNIST, FedMNIST.Parameters(num_clients=NUM_CLIENTS, class_per_client=2, seed=SEED, subset=0.05))

    body_init = CNNMnistBody()
    head_init = CNNMnistHead()
    fedlog_model = CNNMnist(body=body_init, fc1=head_init.fc1, augment=True)
    exp.register_model("FedLog", FedLog, 
            FedLog.Parameters(model=fedlog_model, m=50, num_class=10, num_clients=NUM_CLIENTS,
                lr=0.001, train_batch_size=10, test_batch_size=1000, local_epochs=5,
                seed=SEED, eta_init=None, mean_err=True))
    exp.run()
elif DATASET == "cifar10":
    data.base.VALI_PROPORTION = 0
    NUM_CLIENTS=100
    params = Experiment.Parameters()
    params.show_metric_train = False
    params.global_epochs = 100 
    params.pretrain = True
    params.path = "./res/communication_cost/cifar10/fedlog_"+str(SEED)+"_"
    exp = Experiment(params)
    
    exp.register_dataset(DATASET, FedCIFAR10, FedCIFAR10.Parameters(num_clients=NUM_CLIENTS, class_per_client=2, seed=SEED))
    body_init = CNNCifarBody()
    head_init = CNNCifarHead()
    fedlog_model = CNNCifar(body=body_init, fc2=head_init.fc2, augment=True)
    exp.register_model("FedLog",FedLog,
            FedLog.Parameters(model=fedlog_model, m=100, num_class=10, num_clients=NUM_CLIENTS,
                lr=0.0005, train_batch_size=50, test_batch_size=100, local_epochs=1,
                seed=SEED, eta_init=None, mean_err=True))
    exp.run()
elif DATASET == "cifar100":
    data.base.VALI_PROPORTION = 0
    NUM_CLIENTS=100

    params = Experiment.Parameters()
    params.show_metric_train = False
    params.global_epochs = 150 
    params.pretrain = True
    params.path = "./res/communication_cost/cifar100/fedlog_"+str(SEED)+"_"
    exp = Experiment(params)

    exp.register_dataset(DATASET, FedCIFAR100, FedCIFAR100.Parameters(num_clients=NUM_CLIENTS, class_per_client=10, seed=SEED))
    body_init = CNNCifarBody()
    head_init = CNNCifar100Head()
    fedlog_model = CNNCifar(body=body_init, fc2=head_init.fc2, augment=True)
    exp.register_model("FedLog",FedLog,
            FedLog.Parameters(model=fedlog_model, m=100, num_class=100, num_clients=NUM_CLIENTS,
                lr=0.0005, train_batch_size=50, test_batch_size=100, local_epochs=3,
                seed=SEED, eta_init=None, mean_err=True))
    exp.run()
else:
    raise Exception("Dataset not supported: " + DATASET)

