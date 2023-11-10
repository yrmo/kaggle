"""https://www.kaggle.com/competitions/titanic/data
Variable	Definition	Key
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex	
Age	Age in years	
sibsp	# of siblings / spouses aboard the Titanic	
parch	# of parents / children aboard the Titanic	
ticket	Ticket number	
fare	Passenger fare	
cabin	Cabin number	
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

Variable Notes
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower
age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)
parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.
"""
# %%
import csv
from os import environ
from typing import final

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

RNG = 1337
torch.manual_seed(RNG)

SUBMIT: final = int(environ.setdefault("SUBMIT", "1"))

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
COMBINED: final = pd.concat([train_data.drop("Survived", axis=1), test_data], axis=0)

# for column in combined.columns:
#     if combined[column].isna().any():
#         print(f"{column=} has NaN values")
# column='Age' has NaN values
# column='Fare' has NaN values
# column='Cabin' has NaN values
# column='Embarked' has NaN values

# play to win
MEDIAN_AGE: final = COMBINED.Age.median()
MEDIAN_FARE: final = COMBINED.Fare.median()
# this is funky, you could probably feature number of rooms, it's weird

CABIN_PREFIX_MODE: final = COMBINED.Cabin.mode().item()[0]
CABIN_PREFIX_MAP: final = {
    prefix: i
    for i, prefix in enumerate(
        set([x[0] for x in COMBINED.Cabin.unique().tolist() if x is not np.nan])
    )
}
EMBARKED_MODE: final = COMBINED.Embarked.mode().item()
EMBARKED_MAP: final = {x: i for i, x in enumerate(COMBINED.Embarked.unique().tolist())}
train_data.dropna(subset=["Sex"], inplace=True)

SEX_MAP = {"male": 0, "female": 1}


def clean(df):
    # TODO
    df = df.drop(["Name", "Ticket"], axis=1)
    # survival pclass sex Age sibsp parch ticket fare cabin embarked
    df.Sex = df.Sex.map(SEX_MAP)
    # nan: age fare cabin embarkesd
    df.Age = df.Age.fillna(MEDIAN_AGE)
    df.Fare = df.Fare.fillna(MEDIAN_FARE)
    df.Cabin = (
        df.Cabin.fillna(CABIN_PREFIX_MODE).apply(lambda x: x[0]).map(CABIN_PREFIX_MAP)
    )
    df.Embarked = df.Embarked.fillna(EMBARKED_MODE).map(EMBARKED_MAP)
    return df


# %%
train_data = clean(train_data)
test_data = clean(test_data)

# %%
INPUTS: final = len(test_data.columns)
assert len(train_data.columns) - 1 == len(test_data.columns)
assert len(train_data.select_dtypes(exclude=["int", "float"]).columns) == 0
assert len(test_data.select_dtypes(exclude=["int", "float"]).columns) == 0
print(f"{INPUTS=}")
TEST_COLUMNS: final = test_data.columns.tolist()
TRAIN_COLUMNS: final = train_data.columns.tolist()
print(TEST_COLUMNS, TRAIN_COLUMNS)


# %%
def prepare(df):
    return torch.tensor(df[TEST_COLUMNS].values.astype(float)).float()


if not SUBMIT:
    X = prepare(train_data)
    y = torch.tensor(train_data["Survived"].values.astype(float)).float().view(-1, 1)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RNG
    )
else:
    X_train = prepare(train_data)
    y_train = (
        torch.tensor(train_data["Survived"].values.astype(float)).float().view(-1, 1)
    )


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUTS, INPUTS)
        self.fc2 = nn.Linear(INPUTS, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# %%
model = MLP()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10000):
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
            accuracy = (val_pred == y_val).float().mean()
            print(f"{accuracy=}")

# %%
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

# %%
