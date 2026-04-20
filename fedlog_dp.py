import sys

from utils import setup_seed
from model.fedlog import FedLog
from model.base import * 
from data.regular import *
from expbase import Experiment
import data.base

SEED = int(sys.argv[1])
DATASET = sys.argv[2]
EPSILON = float(sys.argv[3])

setup_seed(SEED)

if DATASET == "cifar10":
    data.base.VALI_PROPORTION = 0
    NUM_CLIENTS=100
    params = Experiment.Parameters()
    params.show_metric_train = False
    params.global_epochs = 100 
    params.pretrain = True
    params.path = "./res/dp/cifar10/fedlog_"+str(EPSILON)+"_"+str(SEED)+"_"
    exp = Experiment(params)
    
    exp.register_dataset(DATASET, FedCIFAR10, FedCIFAR10.Parameters(num_clients=NUM_CLIENTS, class_per_client=2, seed=SEED))
    body_init = CNNCifarBody()
    head_init = CNNCifarHead()
    fedlog_model = CNNCifarDP(b=2,body=body_init, fc2=head_init.fc2, augment=True)
    exp.register_model("FedLog",FedLog,
            FedLog.Parameters(model=fedlog_model, m=100, num_class=10, num_clients=NUM_CLIENTS,
                lr=0.0005, train_batch_size=50, test_batch_size=100, local_epochs=1,
                seed=SEED, eta_init=None, mean_err=True,b=2,dp_epsilon=EPSILON,dp_delta=0.01))
    exp.run()
else:
    raise Exception("Dataset not supported: " + DATASET)

