import csv

import pandas as pd
import torch
import torch.nn as nn


class ZeroNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros(x.size(0), 1)


net = ZeroNet()
data = []
data.append(["PassengerId", "Survived"])
for i in range(len(pd.read_csv("/kaggle/input/titanic/test.csv"))):
    data.append([i + 892, int(net(torch.tensor([1], dtype=torch.float32)).item())])

with open("/kaggle/working/submission.csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    for i, row in enumerate(data):
        csvwriter.writerow(row)

print(pd.read_csv("/kaggle/working/submission.csv"))
