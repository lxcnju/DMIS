import os
import time
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import count_acc, Averager
from utils import append_to_logs

from data import load_classical_data as load_data
from data import MilClassicalDataset as MilDataset
from networks import DMIS

from paths import classical_fdir as fdir


def construct_dataloader(xs, ys, args):
    inds = np.arange(len(xs))
    np.random.shuffle(inds)

    xs = [xs[i] for i in inds]
    ys = [ys[i] for i in inds]

    n_train = int(len(inds) * (1 - args.test_ratio))
    train_xs = [xs[k] for k in range(n_train)]
    train_ys = [ys[k] for k in range(n_train)]

    test_xs = [xs[k] for k in range(n_train, len(inds))]
    test_ys = [ys[k] for k in range(n_train, len(inds))]

    trainset = MilDataset(train_xs, train_ys)
    testset = MilDataset(test_xs, test_ys)

    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True
    )
    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False
    )
    return trainloader, testloader


def train(model, loader, optimizer, args):
    model.train()

    all_scores = []
    for batch_x, batch_bag_y in loader:
        logits, scores = model(batch_x)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, batch_bag_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_scores.append(scores.detach().view(-1).numpy())
    all_scores = np.concatenate(all_scores, axis=0)
    return all_scores


def test(model, loader, args):
    model.eval()

    acc_avg = Averager()

    with torch.no_grad():
        for i, (batch_x, batch_bag_y) in enumerate(loader):
            logits, scores = model(batch_x)

            acc = count_acc(logits, batch_bag_y)
            acc_avg.add(acc)

    acc = acc_avg.item()
    return acc


def main(para_dict):
    print(para_dict)
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    # Local DataLoaders & Iters
    fpath = os.path.join(fdir, "{}.mat".format(args.dataset))
    xs, ys = load_data(fpath)

    trainloader, testloader = construct_dataloader(xs, ys, args)

    model = DMIS(args)

    print(model)
    n_params = sum([
        param.numel() for param in model.parameters()
    ])
    print("Total number of parameters : {}".format(n_params))

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            momentum=args.momentum,
            lr=args.lr, weight_decay=args.l2_decay,
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr, weight_decay=args.l2_decay,
        )
    else:
        raise ValueError("No such optimizer.")

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    all_scores = {}

    for epoch in range(args.epoches):
        scores = train(model, trainloader, optimizer, args)
        acc = test(model, testloader, args)
        print("[Acc: {}] [Tau: {}]".format(acc, model.tau))

        all_scores[epoch] = scores

        model.decay_temperature()
        lr_scheduler.step()

    # record
    fpath = os.path.join("./", args.fname)
    log_name = " ".join([
        "{}:{}".format(k, v) for k, v in para_dict.items()
    ])
    str_time = time.strftime(
        "%Y.%m.%d.%H.%M", time.localtime(time.time())
    )
    str_res = "{}".format(acc)

    logs = [log_name, str_time, str_res]

    append_to_logs(fpath, logs)

    return acc


if __name__ == "__main__":
    dsets = ["musk1", "musk2", "elephant", "fox", "tiger"]
    input_sizes = [166, 166, 230, 230, 230]
    candi_param_dict = {
        "dataset": ["musk1"],
        "input_size": [166],
        "n_classes": [2],
        "test_ratio": [0.2],
        "epoches": [50],
        "batch_size": [1],
        "optimizer": ["Adam"],
        "momentum": [0.9],
        "lr": [0.0003],
        "step_size": [5],
        "gamma": [0.9],
        "l2_decay": [1e-5],
        "init_tau": [5.0],
        "decay": [0.9],
        "min_tau": [0.1],
        "var_norm": ["std"],
        "fname": ["milclassical.log"],
    }

    all_results = {}
    dsets = ["elephant", "fox", "tiger"]
    for d, dset in enumerate(dsets):
        results = {}

        total_acc = 0.0
        n = 5
        for _ in range(n):
            para_dict = {}
            for k, values in candi_param_dict.items():
                para_dict[k] = random.choice(values)

            para_dict["dataset"] = dset
            para_dict["input_size"] = 230

            acc = main(para_dict)
            total_acc += acc

        results = {
            "acc": total_acc / n,
        }
        all_results[dset] = results
    print(all_results)

    for dset, results in all_results.items():
        print("#" * 50)
        print(dset)
        print(results)
        print("#" * 50)
