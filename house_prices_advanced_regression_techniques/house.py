from os import environ
from typing import Final

import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(90053)

SUBMIT: Final = int(environ.setdefault("SUBMIT", "1"))

train_data = pd.read_csv(
    "/kaggle/input/house-prices-advanced-regression-techniques/train.csv"
)
test_data = pd.read_csv(
    "/kaggle/input/house-prices-advanced-regression-techniques/test.csv"
)


class ZeroNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros(x.size(0), 1)


model = ZeroNet()
submission = pd.DataFrame()
submission["Id"] = test_data["Id"]
submission["SalePrice"] = model(torch.tensor(test_data.Id))
submission.to_csv("/kaggle/working/submission.csv", index=False)
