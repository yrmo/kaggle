import csv
from os import environ
from typing import Final

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split  # type: ignore

torch.manual_seed(90053)

SUBMIT: Final = int(environ.setdefault("SUBMIT", "1"))

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

MODE_AGE: Final = train_data.Age.mode().item()

MODE_SEX: Final = train_data.Sex.mode().item()
SEX: Final = {"male": 0, "female": 1}

MODE_EMBARKED: Final = train_data.Embarked.mode().item()
EMBARKED: Final = {"C": 0, "Q": 1, "S": 2}

MEAN_FARE: Final = train_data.Fare.mean().item()

INPUTS: Final = ["Sex", "Pclass", "Age", "SibSp", "Parch", "Embarked", "Cabin", "Fare"]
FEATURES: Final = [
    "Family",  # SibSp + Parch
    # "Deck",  # Cabin prefix letter, nan -> "U" (unused in train/test)
    "AgeMulPclass",
    "FarePerPerson",
]

CABIN_DECK_PREFIX_MAP: Final = {
    x: i
    for i, x in enumerate(
        set(
            [
                x[0]
                for x in filter(
                    lambda x: x is not np.nan,
                    # haha I'm cheating woo!
                    (pd.concat([train_data, test_data], axis=0))
                    .Cabin.unique()
                    .tolist(),
                )
            ]
        )
    )
}
NAN_CABIN_MARKER: Final = "U"
CABIN_DECK_PREFIX_MAP["U"] = max(CABIN_DECK_PREFIX_MAP.values()) + 1  # type: ignore


def pipeline(df: pd.DataFrame) -> torch.Tensor:
    df.Sex.fillna(MODE_SEX, inplace=True)
    df.Age.fillna(MODE_AGE, inplace=True)
    df.Embarked.fillna(MODE_EMBARKED, inplace=True)
    df.Cabin.fillna(NAN_CABIN_MARKER, inplace=True)
    df.Fare.fillna(MEAN_FARE, inplace=True)

    assert "Cabin" in df.columns.tolist()
    df.Cabin = df.Cabin.apply(lambda x: x[0])

    assert all(feature in df.columns.tolist() for feature in ["SibSp", "SibSp"])
    df["Family"] = df["SibSp"] + df["Parch"]

    assert all(feature in df.columns.tolist() for feature in ["Age", "Pclass"])
    df["AgeMulPclass"] = df["Age"] * df["Pclass"]

    assert all(feature in df.columns.tolist() for feature in ["Fare", "Family"])
    df["FarePerPerson"] = df["Fare"] / (df["Family"] + 1)

    assert df[INPUTS].isna().any().any() == False
    assert df[FEATURES].isna().any().any() == False
    for column in FEATURES:
        assert column in df.columns.tolist()

    df.drop(
        list(set(test_data.columns.tolist()) - set(INPUTS + ["PassengerId"])),
        axis=1,
        inplace=True,
    )
    df.Sex = df.Sex.map(SEX)
    df.Embarked = df.Embarked.map(EMBARKED)
    df.Cabin = df.Cabin.map(CABIN_DECK_PREFIX_MAP)

    # from sklearn.preprocessing import StandardScaler
    for column in INPUTS:
        column_mean = df[column].mean()
        column_std = df[column].std()
        df[column] = (df[column] - column_mean) / column_std

    return torch.tensor(df[INPUTS].values.astype(float)).float()


if not SUBMIT:
    X = pipeline(train_data)
    y = torch.tensor(train_data["Survived"].values.astype(float)).float().view(-1, 1)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
else:
    X_train = pipeline(train_data)
    y_train = (
        torch.tensor(train_data["Survived"].values.astype(float)).float().view(-1, 1)
    )


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        N = len(INPUTS) * 8
        self.fc1 = nn.Linear(len(INPUTS), N)
        self.fc2 = nn.Linear(N, N)
        self.fc3 = nn.Linear(N, N)
        self.fc4 = nn.Linear(N, 1)

    def forward(self, x) -> torch.Tensor:
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


model = MLP()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
EPOCHS: Final = 100000

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
                f"titanic: L{round(val_loss.item(), 3)} A{round(val_accuracy.item(), 3)}"
            )

X_test = pipeline(test_data)

with torch.no_grad():
    test_output = model(X_test)
    test_output = (test_output > 0.5).int()

submission_data = list(zip(test_data["PassengerId"], test_output.numpy().flatten()))

with open("/kaggle/working/submission.csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["PassengerId", "Survived"])
    for row in submission_data:
        csvwriter.writerow(row)
