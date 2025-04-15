import torch
from torch import nn


class BaseModel(nn.Module):
    def __init__(self) -> None:
        super(BaseModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor, debug_mode: bool = False) -> torch.Tensor:
        if debug_mode:
            print("Initial shape: ", x.shape)
            # print(x)
        x = self.pool(self.relu(self.conv1(x)))
        if debug_mode:
            print("After first Conv, ReLU, MaxPool: ", x.shape)
        x = self.pool(self.relu(self.conv2(x)))
        if debug_mode:
            print("After second Conv, ReLU, MaxPool: ", x.shape)
        x = self.pool(self.relu(self.conv3(x)))
        if debug_mode:
            print("After third Conv, ReLU, MaxPool: ", x.shape)
        x = self.flatten(x)
        if debug_mode:
            print("After flattening: ", x.shape)
        x = self.dropout(x)
        if debug_mode:
            print("After dropout: ", x.shape)
        x = self.relu(self.fc1(x))
        if debug_mode:
            print("After FC1 and ReLU: ", x.shape)
        x = self.dropout(x)
        if debug_mode:
            print("After dropout: ", x.shape)
        x = self.fc2(x)
        x = self.log_softmax(x)
        if debug_mode:
            print("After FC2 and LogSoftmax: ", x.shape)
            print(x)
        return x


class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # Adjusted input size
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.log_softmax(self.fc2(x))
        return x


class SmallStudentModel(nn.Module):
    def __init__(self):
        super(SmallStudentModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)  # Adjusted input size
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.log_softmax(self.fc2(x))
        return x
