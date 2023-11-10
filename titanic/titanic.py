import csv
from os import environ

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

torch.manual_seed(1337)

SUBMIT = int(environ.setdefault("SUBMIT", "1"))

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.dropna(subset=["Sex"], inplace=True)


def clean(df):
    df["Sex"] = df["Sex"].apply(lambda x: 0 if x == "male" else 1)
    return df


def prepare(df):
    return torch.tensor(df[["Sex", "Pclass"]].values.astype(float)).float()


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
        self.fc1 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return x


model = MLP()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if not SUBMIT and epoch % 100 == 0:
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            val_pred = (val_output > 0.5).float()
            val_accuracy = (val_pred == y_val).float().mean()
            print(f"{val_loss=}", f"{val_accuracy=}")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
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
