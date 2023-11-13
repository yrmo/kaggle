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

if "LotFrontage" in INPUTS:
    MEDIAN_LOT_FRONTAGE = train_data["LotFrontage"].median().item()

y_train = torch.tensor(train_data[["SalePrice"]].to_numpy(), dtype=torch.float32)
X_train = train_data[INPUTS].copy()
if "LotFrontage" in INPUTS:
    X_train.fillna(MEDIAN_LOT_FRONTAGE, inplace=True)
print(X_train)
for input in INPUTS:
    # fixes exploding gradients!
    X_train[input] = (X_train[input] - train_data[input].mean()) / train_data[
        input
    ].std()
assert X_train.isna().any().any() == False
X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
print(X_train)
print(f"{y_train.size()=}", f"{X_train.size()=}")
assert X_train is not None
assert y_train is not None
assert type(X_train) == torch.Tensor
assert type(y_train) == torch.Tensor


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        N = len(INPUTS)
        self.fc1 = nn.Linear(N, N)
        self.fc2 = nn.Linear(N, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = MLP()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.000005)

epochs, losses = [], []
best_loss = float("inf")
best_epoch = 0

EPOCHS = 500000
PATIENCE = 0
CHECKPOINTS = 15
for epoch in range(EPOCHS):
    if PATIENCE > 3:
        print(f"{PATIENCE=}")
        break
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    if epoch % (EPOCHS // CHECKPOINTS) == 0 and loss.item() < best_loss:
        best_loss = loss.item()
        best_epoch = epoch
        PATIENCE = 0
    elif epoch % (EPOCHS // CHECKPOINTS) == 0:
        PATIENCE += 1
    loss.backward()
    optimizer.step()
    if epoch % (EPOCHS // CHECKPOINTS) == 0:
        epochs.append(epoch)
        losses.append(loss.item())
        print(f"house-prices: L{loss.item()}")

print(f"{best_loss=}", f"{best_epoch=}")

submission = pd.DataFrame()
submission["Id"] = test_data.Id
with torch.no_grad():
    X_test = test_data[INPUTS].copy()
    if "LotFrontage" in INPUTS:
        X_test.fillna(MEDIAN_LOT_FRONTAGE, inplace=True)
    for input in INPUTS:
        # fixes exploding gradients!
        X_test[input] = (X_test[input] - test_data[input].mean()) / test_data[
            input
        ].std()
    submission["SalePrice"] = model(
        torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    )
submission.to_csv("/kaggle/working/submission.csv", index=False)

if not SUBMIT:
    plt.plot(epochs, losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./house_prices/LossVersusEpoch.png")
    plt.show()
