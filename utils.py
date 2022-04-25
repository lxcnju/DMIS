import os
import pandas as pd

import torch
import torch.nn as nn

column_names = ["id", "label", "data"]

try:
    import moxing as mox

    def read_file(path):
        with mox.file.File(path, 'r') as fr:
            da_df = pd.read_csv(
                fr, index_col=False, header=None, names=column_names)
        return da_df

    def save_data(da_df, path):
        with mox.file.File(path, 'w') as fr:
            da_df.to_csv(fr)
        print("File saved in {}.".format(path))

    def append_to_logs(fpath, logs):
        with mox.file.File(fpath, "a") as fa:
            for log in logs:
                fa.write("{}\n".format(log))
            fa.write("\n")

except Exception:
    def read_file(path):
        da_df = pd.read_csv(
            path, index_col=False, header=None, names=column_names)
        return da_df

    def save_data(da_df, path):
        da_df.to_csv(path)
        print("File saved in {}.".format(path))

    def append_to_logs(fpath, logs):
        with open(fpath, "a", encoding="utf-8") as fa:
            for log in logs:
                fa.write("{}\n".format(log))
            fa.write("\n")


def listfiles(fdir):
    for root, dirs, files in os.walk(fdir):
        print(root, dirs, files)


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

def analyse_distributions(class_distributions):
    for scene in ["scene1", "scene2"]:
        part1_classes = sorted(
            list(class_distributions[scene]["part1"].keys())
        )
        part2_classes = sorted(
            list(class_distributions[scene]["part2"].keys())
        )
        part3_classes = sorted(
            list(class_distributions[scene]["part3"].keys())
        )
        test_classes = sorted(
            list(class_distributions[scene]["test"].keys())
        )
        print(part1_classes)
        print(part2_classes)
        print(part3_classes)
        print(test_classes)


if __name__ == "__main__":
    listfiles(fdir="./")

    from config import scene_paths
    fpath = scene_paths["scene1"]["part1"]
    df = read_file(fpath)
    print(df.columns)
    print(df.iloc[0:5])
