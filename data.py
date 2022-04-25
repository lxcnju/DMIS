import copy
import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils import data


def load_classical_data(fpath):
    infos = sio.loadmat(fpath)
    print(infos.keys())
    
    bag_ids = np.array(infos["bag_ids"]).reshape(-1)
    features = np.array(infos["features"].todense())
    ins_labels = np.array(infos["labels"].todense()).reshape(-1)
    
    data = []
    labels = []
    for i in np.unique(bag_ids):
        indx = np.argwhere(bag_ids == i).reshape(-1)
        data.append(features[indx])
        
        label = np.unique(ins_labels[indx])
        if label == -1.0:
            label = 0.0
        else:
            label = 1.0
        labels.append(label)
        
    return data, labels
    

class MilClassicalDataset(data.Dataset):
    def __init__(self, data, labels):
        self.data = copy.deepcopy(data)
        self.labels = copy.deepcopy(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, k):
        x = self.data[k]
        y = self.labels[k]
        
        x = torch.FloatTensor(x)
        y = torch.LongTensor([y])[0]
        return x, y


def load_mil_text_data(fpath):
    infos = sio.loadmat(fpath)["data"]
    data = []
    bag_labels = []
    ins_labels = []
    for info in infos:
        data.append(info[0])
        bag_labels.append(info[1])
        ins_labels.append(info[2])
    return data, bag_labels, ins_labels



class MilTextDataset(data.Dataset):
    def __init__(self, data, bag_labels, ins_labels):
        self.data = copy.deepcopy(data)
        self.bag_labels = copy.deepcopy(bag_labels)
        self.ins_labels = copy.deepcopy(ins_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, k):
        x = self.data[k]
        bag_y = self.bag_labels[k].reshape((-1, ))
        ins_y = self.ins_labels[k].reshape((-1, ))

        x = torch.FloatTensor(x)
        bag_y = torch.LongTensor(bag_y)[0]
        ins_y = torch.LongTensor(ins_y)
        return x, bag_y, ins_y


if __name__ == "__main__":
    """
    fdir = r"C:/workspace/datasets/mil-text-data/data"
    fname = "sci.crypt"
    fpath = os.path.join(fdir, "{}.mat".format(fname))
    data, bag_labels, ins_labels = load_mil_text_data(fpath)
    print(np.unique(bag_labels))
    print(ins_labels[0])
    """
    
    fdir = r"C:/workspace/datasets/mil-classical-data/"
    fname = "tiger"
    fpath = os.path.join(fdir, "{}.mat".format(fname))
    data, labels = load_classical_data(fpath)
    print(np.unique(labels))
    print(data[0].shape)


