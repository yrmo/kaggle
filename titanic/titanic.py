import csv
from os import environ
from typing import final

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

torch.manual_seed(1337)

SUBMIT = int(environ.setdefault("SUBMIT", "1"))

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
MODE_SEX: final = train_data.Sex.mode().item()

SEX: final = {"male": 0, "female": 1}

INPUTS: final = ["Sex", "Pclass"]


def clean(df):
    df.Sex.fillna(MODE_SEX, inplace=True)
    df.drop(list(set(test_data.columns.tolist()) - set(INPUTS)), axis=1)
    df.Sex = df.Sex.map(SEX)
    return df


def prepare(df):
    return torch.tensor(df[INPUTS].values.astype(float)).float()


train_data = clean(train_data)

if not SUBMIT:
    X = prepare(train_data)
    y = torch.tensor(train_data["Survived"].values.astype(float)).float().view(-1, 1)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
else:
    X_train = prepare(train_data)
    y_train = (
        torch.tensor(train_data["Survived"].values.astype(float)).float().view(-1, 1)
    )


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        N = len(INPUTS) * 3
        self.fc1 = nn.Linear(len(INPUTS), N)
        self.fc2 = nn.Linear(N, N)
        self.fc3 = nn.Linear(N, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


model = MLP()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
EPOCHS: final = 50000

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if not SUBMIT and epoch % (EPOCHS // 10) == 0:
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            val_pred = (val_output > 0.5).float()
            val_accuracy = (val_pred == y_val).float().mean()
            print(
                f"titanic: L{round(val_loss.item(), 2)} A{round(val_accuracy.item(), 2)}"
            )

test_data = clean(test_data)
X_test = prepare(test_data)

with torch.no_grad():
    test_output = model(X_test)
    test_output = (test_output > 0.5).int()

submission_data = list(zip(test_data["PassengerId"], test_output.numpy().flatten()))

with open("/kaggle/working/submission.csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["PassengerId", "Survived"])
    for row in submission_data:
        csvwriter.writerow(row)
