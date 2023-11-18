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
print(TRAIN)

FEATURES = [
    "LotArea",
    "LotFrontage",
]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        N = len(FEATURES)
        self.fc1 = nn.Linear(N, N)
        self.fc2 = nn.Linear(N, N)
        self.fc3 = nn.Linear(N, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


def clean(df):
    df.fillna(TRAIN.LotFrontage.mean(), inplace=True)
    return df


def normalize(df: pd.DataFrame, features: list[str]) -> torch.Tensor:
    df = df[features].copy()
    assert df.isna().any().any() == False
    for column in df.columns:
        df[column] = (TRAIN[column] - TRAIN[column].mean()) / TRAIN[column].std()
    return torch.tensor(df.to_numpy(), dtype=torch.float32)


def unnormalize(t: torch.Tensor) -> torch.Tensor:
    t = (t * TRAIN.SalePrice.std().item()) + TRAIN.SalePrice.mean().item()
    return torch.round(t)


X = normalize(clean(TRAIN), FEATURES)
y = normalize(TRAIN, ["SalePrice"])

epochs = []
losses = []

model = Model()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
EPOCHS: Final = 1000
for epoch in range(EPOCHS + 1):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    epochs.append(epoch)
    losses.append(loss.item())

    if epoch % (EPOCHS // 10) == 0:
        print(f"house: E{epoch} L{loss.item()} EX{unnormalize(model(X)[0]).item()}")

if not SUBMIT:
    plt.plot(epochs, losses)
    plt.title("Kaggle House Price Prediction")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("./house/house.png")
    plt.show()
