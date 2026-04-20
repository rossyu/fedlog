import sys

from utils import setup_seed
from model.csfl import CSFL
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
    params.path = "./res/communication_cost/mnist/csfl_"+str(SEED)+"_"
    exp = Experiment(params)

    exp.register_dataset(DATASET, FedMNIST, FedMNIST.Parameters(num_clients=NUM_CLIENTS, class_per_client=2, seed=SEED, subset=0.05))

    body_init = CNNMnistBody()
    head_init = CNNMnistHead()
    csfl_model = CNNMnist(body=body_init, fc1=head_init.fc1, fc2=head_init.fc2, augment=False)
    exp.register_model("CS-FL", CSFL,
           CSFL.Parameters(global_model=csfl_model,
            num_class=10, num_clients=NUM_CLIENTS, lr=0.001, train_batch_size=10,
            test_batch_size=1000, local_epochs=5, sparcity=0.005, reduced_dim=0.1,
            lr_multiplier=1, binary=False, seed=SEED, mean_err=True)) 
    exp.run()
elif DATASET == "cifar10":
    data.base.VALI_PROPORTION = 0
    NUM_CLIENTS=100
    params = Experiment.Parameters()
    params.show_metric_train = False
    params.global_epochs = 100 
    params.pretrain = True
    params.path = "./res/communication_cost/cifar10/csfl_"+str(SEED)+"_"
    exp = Experiment(params)

    exp.register_dataset(DATASET, FedCIFAR10, FedCIFAR10.Parameters(num_clients=NUM_CLIENTS, class_per_client=2, seed=SEED))
    body_init = CNNCifarBody()
    head_init = CNNCifarHead()
    csfl_model = CNNCifar(body=body_init, fc2=head_init.fc2, fc3=head_init.fc3, augment=False)
    exp.register_model("CS-FL", CSFL,
           CSFL.Parameters(global_model=csfl_model,
            num_class=10, num_clients=NUM_CLIENTS, lr=0.001, train_batch_size=10,
            test_batch_size=1000, local_epochs=1, sparcity=0.0005, reduced_dim=0.1,
            lr_multiplier=10, binary=False, seed=SEED, mean_err=True))
    exp.run()
elif DATASET == "cifar100":
    data.base.VALI_PROPORTION = 0
    NUM_CLIENTS=100

    params = Experiment.Parameters()
    params.show_metric_train = False
    params.global_epochs = 100 
    params.pretrain = True
    params.path = "./res/communication_cost/cifar100/csfl_"+str(SEED)+"_"
    exp = Experiment(params)

    exp.register_dataset(DATASET, FedCIFAR100, FedCIFAR100.Parameters(num_clients=NUM_CLIENTS, class_per_client=10, seed=SEED))
    body_init = CNNCifarBody()
    head_init = CNNCifar100Head()
    csfl_model = CNNCifar(body=body_init, fc2=head_init.fc2, fc3=head_init.fc3, augment=False)
    exp.register_model("CS-FL", CSFL,
           CSFL.Parameters(global_model=csfl_model,
            num_class=100, num_clients=NUM_CLIENTS, lr=0.001, train_batch_size=10,
            test_batch_size=1000, local_epochs=1, sparcity=0.0005, reduced_dim=0.1,
            lr_multiplier=10, binary=False, seed=SEED, mean_err=True))
    exp.run()
else:
    raise Exception("Dataset not supported: " + DATASET)

