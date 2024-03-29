import csv
import re
from os import environ
from typing import Final

import matplotlib.pyplot as plt  # type: ignore
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

INPUTS: Final = [
    "Sex",
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Embarked",
    "Cabin",
    "Fare",
    "Name",
    "Ticket",
]
FEATURES: Final = [
    "Family",  # SibSp + Parch
    "AgeMulPclass",
    "FarePerPerson",
    "Child",
    "FamilySurvived",
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

# honors with at least two examples (exclude... Countess., Capt.,...)
HONORS: Final = [
    "Mr.",
    "Mrs.",
    "Miss.",
    "Master.",
    "Dr.",
    "Col.",
    "Mlle.",
    "Rev.",
    "Major.",
]
HONORS_MISC_MARKER: Final = "Misc."
HONORS_MAP: Final = {honor: i for i, honor in enumerate(HONORS + [HONORS_MISC_MARKER])}
# [name for name in train_data.Name.tolist() if all(honor not in name for honor in [HONORS])]

TICKET: Final = ["A", "P", "S", "F", "W", "C"]
TICKET_MAP: Final = {letter: i for i, letter in enumerate(TICKET)}
NUMERIC_TICKET_MARKER: Final = "N"
TICKET_MAP[NUMERIC_TICKET_MARKER] = max(TICKET_MAP.values()) + 1  # type: ignore
# [name for name in train_data.Ticket.tolist() if all(honor not in name for honor in TICKET)]


def get_families_who_all_survived():
    train_data["Surname"] = train_data["Name"].apply(lambda x: x.split(",")[0])
    grouped_data = train_data.groupby("Surname")
    survival_rates = grouped_data["Survived"].mean()
    family_sizes = grouped_data.size()
    families_all_survived = survival_rates[(survival_rates == 1) & (family_sizes > 1)]
    return {surname: family_sizes[surname] for surname in families_all_survived.index}


WHOLE_FAMILY_SURVIVED: Final = list(get_families_who_all_survived().keys())


def pipeline(df: pd.DataFrame) -> torch.Tensor:
    df.Sex.fillna(MODE_SEX, inplace=True)
    df.Age.fillna(MODE_AGE, inplace=True)
    df.Embarked.fillna(MODE_EMBARKED, inplace=True)
    df.Cabin.fillna(NAN_CABIN_MARKER, inplace=True)
    df.Fare.fillna(MEAN_FARE, inplace=True)

    df["FamilySurvived"] = df.Name.apply(
        lambda name: 1 if name.split(",")[0] in WHOLE_FAMILY_SURVIVED else 0
    )

    assert "Name" in df.columns.tolist()
    df["Child"] = df.Name.apply(lambda name: 1 if "Master." in name else 0)

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
        list(
            set(test_data.columns.tolist()) - set(INPUTS + FEATURES + ["PassengerId"])
        ),
        axis=1,
        inplace=True,
    )
    df.Sex = df.Sex.map(SEX)
    df.Embarked = df.Embarked.map(EMBARKED)
    df.Cabin = df.Cabin.map(CABIN_DECK_PREFIX_MAP)

    def replace_name_with_honor(name: str) -> str:
        for honor in HONORS:
            if honor in name:
                return honor
        return HONORS_MISC_MARKER

    df.Name = df.Name.apply(replace_name_with_honor)
    df.Name = df.Name.map(HONORS_MAP)

    def replace_ticket_with_letter(ticket: str) -> str:
        for letter in TICKET:
            if letter in ticket:
                return letter
        return NUMERIC_TICKET_MARKER

    df.Ticket = df.Ticket.apply(replace_ticket_with_letter)
    df.Ticket = df.Ticket.map(TICKET_MAP)

    # from sklearn.preprocessing import StandardScaler
    for column in INPUTS:
        column_mean = df[column].mean()
        column_std = df[column].std()
        df[column] = (df[column] - column_mean) / column_std

    return torch.tensor(df[INPUTS + FEATURES].values.astype(float)).float()


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
        N = len(INPUTS + FEATURES)
        self.fc1 = nn.Linear(len(INPUTS + FEATURES), N)
        self.fc2 = nn.Linear(N, N * 2)
        self.fc3 = nn.Linear(N * 2, N)
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
EPOCHS: Final = 35000

val_losses = []
losses = []
epochs = []

PATIENCE: Final = 2
best_loss = float("inf")
counter = 0
best_model_state = None
best_model_epoch = None

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % (EPOCHS // 10) == 0:
        print(f"{round(loss.item(), 3)}")

    if not SUBMIT and epoch % (EPOCHS // 10) == 0:
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            val_pred = (val_output > 0.5).float()
            val_accuracy = (val_pred == y_val).float().mean()

            print(
                f"titanic: E{epoch} L{round(val_loss.item(), 3)} A{round(val_accuracy.item(), 3)}"
            )

            losses.append(loss.item())
            epochs.append(epoch)
            val_losses.append(val_loss.item())

            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                counter = 0
                best_model_state = model.state_dict()
                best_model_epoch = epoch
            else:
                counter += 1

            if counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

if not SUBMIT:
    print(f"Best model state @ epoch {best_model_epoch}")
    model.load_state_dict(best_model_state)

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

if not SUBMIT:
    plt.plot(epochs, losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.axvline(x=best_model_epoch, color="r", linestyle="--", label="Best Model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.savefig("./titanic/LossVersusEpoch.png")
    plt.show()
