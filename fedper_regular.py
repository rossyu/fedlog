import sys

from utils import setup_seed
from model.fedper import FedPer
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
    params.path = "./res/communication_cost/mnist/fedper_"+str(SEED)+"_"
    exp = Experiment(params)

    exp.register_dataset(DATASET, FedMNIST, FedMNIST.Parameters(num_clients=NUM_CLIENTS, class_per_client=2, seed=SEED, subset=0.05))

    body_init = CNNMnistBody()
    head_init = CNNMnistHead()
    fedper_model = CNNMnist(body=None, fc1=head_init.fc1, fc2=head_init.fc2, augment=False)
    exp.register_model("FedPer", FedPer, 
            FedPer.Parameters(local_model=fedper_model, global_model=body_init,
                num_class=10, num_clients=NUM_CLIENTS,
                lr=0.001, train_batch_size=10, test_batch_size=1000, local_epochs=5,
                seed=SEED, mean_err=True))
    exp.run()
elif DATASET == "cifar10":
    data.base.VALI_PROPORTION = 0
    NUM_CLIENTS=100
    params = Experiment.Parameters()
    params.show_metric_train = False
    params.global_epochs = 100
    params.pretrain = True
    params.path = "./res/communication_cost/cifar10/fedper_"+str(SEED)+"_"
    exp = Experiment(params)

    exp.register_dataset(DATASET, FedCIFAR10, FedCIFAR10.Parameters(num_clients=NUM_CLIENTS, class_per_client=2, seed=SEED))
    body_init = CNNCifarBody()
    head_init = CNNCifarHead()
    fedper_model = CNNCifar(body=None, fc2=head_init.fc2, fc3=head_init.fc3, augment=False)
    exp.register_model("FedPer", FedPer,
            FedPer.Parameters(local_model=fedper_model, global_model=body_init,
                num_class=10, num_clients=NUM_CLIENTS,
                lr=0.0005, train_batch_size=50, test_batch_size=100, local_epochs=1,
                seed=SEED, mean_err=True))
    exp.run()
elif DATASET == "cifar100":
    data.base.VALI_PROPORTION = 0
    NUM_CLIENTS=100

    params = Experiment.Parameters()
    params.show_metric_train = False
    params.global_epochs = 150 
    params.pretrain = True
    params.path = "./res/communication_cost/cifar100/fedper_"+str(SEED)+"_"
    exp = Experiment(params)

    exp.register_dataset(DATASET, FedCIFAR100, FedCIFAR100.Parameters(num_clients=NUM_CLIENTS, class_per_client=10, seed=SEED))
    body_init = CNNCifarBody()
    head_init = CNNCifar100Head()
    fedper_model = CNNCifar(body=None, fc2=head_init.fc2, fc3=head_init.fc3, augment=False)
    exp.register_model("FedPer", FedPer,
            FedPer.Parameters(local_model=fedper_model, global_model=body_init,
                num_class=100, num_clients=NUM_CLIENTS,
                lr=0.0005, train_batch_size=50, test_batch_size=100, local_epochs=3,
                seed=SEED, mean_err=True))
    exp.run()
else:
    raise Exception("Dataset not supported: " + DATASET)
