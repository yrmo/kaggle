from os import environ
from typing import Final

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(90053)

SUBMIT: Final = int(environ.setdefault("SUBMIT", "1"))

TRAIN: Final = pd.read_csv(
    "/kaggle/input/house-prices-advanced-regression-techniques/train.csv"
)
TEST: Final = pd.read_csv(
    "/kaggle/input/house-prices-advanced-regression-techniques/test.csv"
)

FEATURES = [
    "",
]

print(TRAIN)