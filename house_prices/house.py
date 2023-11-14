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

train_data = pd.read_csv(
    "/kaggle/input/house-prices-advanced-regression-techniques/train.csv"
)
test_data = pd.read_csv(
    "/kaggle/input/house-prices-advanced-regression-techniques/test.csv"
)

INPUTS = [
    "LotFrontage",  # nan
    "LotArea",
    "OverallQual",
    # "OverallCond",  # no improv. A/B tested w/ LotFrontage, LotArea, OverallQual
]

MEAN_INPUTS = train_data[INPUTS].mean().copy()
STD_INPUTS = train_data[INPUTS].std().copy()

if "LotFrontage" in INPUTS:
    MEDIAN_LOT_FRONTAGE = train_data["LotFrontage"].median().item()

y_train = torch.tensor(train_data[["SalePrice"]].to_numpy(), dtype=torch.float32)
assert y_train is not None
assert type(y_train) == torch.Tensor
print(f"{y_train.size()=}")


def pipeline(df: pd.DataFrame) -> torch.Tensor:
    X = df[INPUTS].copy()
    if "LotFrontage" in INPUTS:
        X.fillna(MEDIAN_LOT_FRONTAGE, inplace=True)
    print(X)
    # fixes exploding gradients!
    X = (X - MEAN_INPUTS) / STD_INPUTS
    assert X.isna().any().any() == False
    return X


X_train = torch.tensor(pipeline(train_data).to_numpy(), dtype=torch.float32)
print(X_train)
print(f"{X_train.size()=}")
assert X_train is not None
assert type(X_train) == torch.Tensor


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        N = len(INPUTS)
        self.fc1 = nn.Linear(len(INPUTS), N)
        self.fc2 = nn.Linear(N, N)
        self.fc3 = nn.Linear(N, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x


model = Model()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.00000005)

epochs, losses = [], []
best_loss = float("inf")
best_epoch = 0
best_model = None

EPOCHS = 10000
CHECKPOINTS = 15
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    if epoch % (EPOCHS // CHECKPOINTS) == 0 and loss.item() < best_loss:
        best_loss = loss.item()
        best_epoch = epoch
        best_model = model.state_dict()
    loss.backward()
    optimizer.step()
    if epoch % (EPOCHS // CHECKPOINTS) == 0:
        epochs.append(epoch)
        losses.append(loss.item())
        print(f"house-prices: E{epoch} L{loss.item()}")

print(f"{best_epoch=}", f"{best_loss=}")
model.load_state_dict(best_model)

submission = pd.DataFrame()
submission["Id"] = test_data.Id
with torch.no_grad():
    X_test = torch.tensor(pipeline(test_data).to_numpy(), dtype=torch.float32)
    print(model(X_test))
    submission["SalePrice"] = model(X_test)
submission.to_csv("/kaggle/working/submission.csv", index=False)

if not SUBMIT:
    plt.plot(epochs, losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./house_prices/LossVersusEpoch.png")
    plt.show()
