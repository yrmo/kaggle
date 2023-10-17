import csv

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.dropna(subset=["Sex"], inplace=True)
train_data["Sex"] = train_data["Sex"].apply(lambda x: 0 if x == "male" else 1)

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return x


model = SimpleNet()

X_train = (
    torch.tensor(train_data[["Sex", "Pclass"]].values.astype(float)).float().view(-1, 2)
)
y_train = torch.tensor(train_data["Survived"].values.astype(float)).float().view(-1, 1)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for i, epoch in enumerate(range(1000)):
    if i % 100 == 0:
        print(f"{epoch=}")
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

test_data["Sex"] = test_data["Sex"].apply(lambda x: 0 if x == "male" else 1)
X_test = (
    torch.tensor(test_data[["Sex", "Pclass"]].values.astype(float)).float().view(-1, 2)
)

with torch.no_grad():
    test_output = model(X_test)
    test_output = (test_output > 0.5).int()

submission_data = list(zip(test_data["PassengerId"], test_output.numpy().flatten()))

with open("/kaggle/working/submission.csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["PassengerId", "Survived"])
    for row in submission_data:
        csvwriter.writerow(row)
