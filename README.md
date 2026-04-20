# FedLog

FedLog is a communication efficient personalized federated classification algorithm. It is published with TMLR under the name [FedLog: Personalized Federated Classification with Less Communication and More Flexibility](https://arxiv.org/abs/2407.08337). Please cite this paper if you use FedLog.

## Setup

To setup the environment of FedLog, execute
```
conda create --name <env> --file requirements.txt
```

## Run experiments

To run experiments in Sec. 4.2, execute
```
python <algo_name>_regular.py <seed> <dataset>
```
where dataset is any of mnist, cifar10, cifar100

To run experiments in Sec. 4.4, execute
```
python <algo_name>_flex.py <seed> cifar10
```

To run experiments in Sec. 4.5, execute
```
python <algo_name>_dp.py <seed> cifar10 <epsilon>
```

Results will be saved in the folder *res/\<experiment\>/\<dataset\>*.
