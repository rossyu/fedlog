import sys

from utils import setup_seed
from model.lgfedavgflex import LGFedAvg
from model.base import * 
from data.regular import *
from expbase import Experiment
import data.base

SEED = int(sys.argv[1])
DATASET = sys.argv[2]

setup_seed(SEED)

def build_cifar10_body1():
    body_init = CNNCifarBody()
    head_init = CNNCifarHead()
    model = CNNCifar(body=body_init, fc2=head_init.fc2, augment=False)
    return model
def build_cifar10_body2():
    body_init = CNNCifarBody(100)
    model = CNNCifar(body=body_init, augment=False)
    return model

if DATASET == "cifar10":
    data.base.VALI_PROPORTION = 0
    NUM_CLIENTS=100
    params = Experiment.Parameters()
    params.show_metric_train = False
    params.global_epochs = 100 
    params.pretrain = True
    params.path = "./res/flex/cifar10/lgfedavg1_"+str(SEED)+"_"
    exp = Experiment(params)

    exp.register_dataset(DATASET, FedCIFAR10, FedCIFAR10.Parameters(num_clients=NUM_CLIENTS, class_per_client=2, seed=SEED))
    head_init = CNNCifarHead()
    exp.register_model("LG-FedAvg 1", LGFedAvg,
            LGFedAvg.Parameters(local_model=[build_cifar10_body1, build_cifar10_body2],
                global_model=head_init.fc3, num_class=10, num_clients=NUM_CLIENTS, lr=0.0005, train_batch_size=50,
                test_batch_size=100, local_epochs=1, avg_epochs=0, seed=SEED, mean_err=True))
    exp.run()
else:
    raise Exception("Dataset not supported: " + DATASET)

