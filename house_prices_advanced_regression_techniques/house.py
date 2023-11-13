from os import environ
from typing import Final

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(90053)

SUBMIT: Final = int(environ.setdefault("SUBMIT", "1"))

train_data = pd.read_csv(
    "/kaggle/input/house-prices-advanced-regression-techniques/train.csv"
)
test_data = pd.read_csv(
    "/kaggle/input/house-prices-advanced-regression-techniques/test.csv"
)

INPUTS = ["LotArea"]

y_train = torch.tensor(train_data[["SalePrice"]].to_numpy(), dtype=torch.float32)
X_train = torch.tensor(train_data[INPUTS].to_numpy(), dtype=torch.float32)
X_train = (X_train - X_train.mean()) / X_train.std()  # fixes exploding gradients!

print(f"{y_train.size()=}", f"{X_train.size()=}")
assert X_train is not None
assert y_train is not None
assert type(X_train) == torch.Tensor
assert type(y_train) == torch.Tensor


class Neuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(len(INPUTS), 1)

    def forward(self, x):
        x = self.fc1(x)
        return x


model = Neuron()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

epochs, losses = [], []

EPOCHS = 3000
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if epoch % (EPOCHS // 10) == 0:
        epochs.append(epoch)
        losses.append(loss.item())
        print(f"{loss.item()=}")

submission = pd.DataFrame()
submission["Id"] = test_data.Id
with torch.no_grad():
    submission["SalePrice"] = model(
        torch.tensor(test_data[INPUTS].to_numpy(), dtype=torch.float32)
    )
submission.to_csv("/kaggle/working/submission.csv", index=False)

if not SUBMIT:
    plt.plot(epochs, losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./house_prices_advanced_regression_techniques/LossVersusEpoch.png")
    plt.show()
