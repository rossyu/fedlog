import abc
import numpy as np

VALI_PROPORTION = 1/5

class Data(abc.ABC):
    def __init__(self):
        self.trainset = None
        self.testset = None

    # returns train_index, test_index, vali_index
    # if test_size or validation_size are invalid, corresponding returns should be None
    @abc.abstractmethod
    def read_and_split(self):
        print("Should never reach here")

def iid(dataset, num_users, seed=None):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    rng = np.random.default_rng(seed=seed)
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        chosen = set(rng.choice(all_idxs, num_items, replace=False))
        dict_users[i] = list(chosen)
        all_idxs = list(set(all_idxs) - chosen)
    return dict_users

def noniid(dataset, num_users, shard_per_user, rand_set_all=[], seed=None, subset=np.inf):
    rng = np.random.default_rng(seed=seed)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    is_train = (len(rand_set_all) == 0) and VALI_PROPORTION > 0
    if is_train:
        dict_vali_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    length = int(min(len(dataset), subset))
    for i in range(length):
        label = dataset.targets[i].item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets[:length]))
    shard_per_class = int(shard_per_user * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        rng.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    print("vali_proportion =", VALI_PROPORTION)
    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        rand_set_vali = []
        for label in rand_set_label:
            idx = rng.choice(len(idxs_dict[label]), replace=False)
            train_vali = idxs_dict[label].pop(idx)
            if is_train:
                rng.shuffle(train_vali)
                bound = int(len(train_vali)*VALI_PROPORTION)
                rand_set.append(train_vali[bound:])
                rand_set_vali.append(train_vali[:bound])
            else:
                rand_set.append(train_vali)
        dict_users[i] = np.concatenate(rand_set)
        if is_train:
            dict_vali_users[i] = np.concatenate(rand_set_vali)

    test = []
    for key, value in dict_users.items():
        x = np.unique((dataset.targets)[value])
        assert(len(x)) <= shard_per_user
        test.append(value)
    if is_train:
        for key, value in dict_vali_users.items():
            x = np.unique((dataset.targets)[value])
            assert(len(x)) <= shard_per_user
            test.append(value)
    test = np.concatenate(test)
    assert(len(test) == length)
    assert(len(set(list(test))) == length)
    #assert(len(test) == len(dataset))
    #assert(len(set(list(test))) == len(dataset))

    if is_train:
        return [dict_users, dict_vali_users], rand_set_all
    else:
        return [dict_users, None], rand_set_all
