from os import environ
from typing import Final

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split  # type: ignore

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
    x = df.copy()
    assert x[FEATURES].isna().any().any() == True
    x["LotFrontage"] = x["LotFrontage"].fillna(TRAIN["LotFrontage"].mean())
    assert x[FEATURES].isna().any().any() == False
    return x


def normalize(df: pd.DataFrame, features: list[str]) -> torch.Tensor:
    df = df[features].copy()
    assert df[features].isna().any().any() == False
    for feature in features:
        df[feature] = (clean(TRAIN)[feature] - clean(TRAIN)[feature].mean()) / clean(
            TRAIN
        )[feature].std()
    return torch.tensor(df[features].to_numpy(), dtype=torch.float32)


def unnormalize(t: torch.Tensor) -> torch.Tensor:
    t = (t * TRAIN.SalePrice.std()) + TRAIN.SalePrice.mean()
    return torch.round(t)


X = normalize(clean(TRAIN), FEATURES)
y = normalize(TRAIN, ["SalePrice"])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=90053
)

epochs = []
losses = []
val_losses = []

model = Model()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
EPOCHS: Final = 100
for epoch in range(EPOCHS + 1):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if epoch % (EPOCHS // 10) == 0:
        epochs.append(epoch)
        losses.append(loss.item())
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val)
        print(
            f"house: E{epoch} L{loss.item()} V{val_loss.item()} EX{unnormalize(model(X)[0]).item()}"
        )
        val_losses.append(val_loss.item())


if not SUBMIT:
    plt.plot(epochs, losses, epochs, val_losses)
    plt.title("Kaggle House Price Prediction")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("./house/house.png")
    plt.show()
