import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPModel, self).__init__()
        """
        Just to see how would it work with a simple MLP model: Train 100% Acc while test Acc is around 75%(?)
        """
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class MLPMapper(nn.Module):
    def __init__(self, input_dim, interim_dim, output_dim):
        super(MLPMapper, self).__init__()
        """
        Just to see how would it work with a simple MLP model: Train 100% Acc while test Acc is around 75%(?)
        """
        self.fc1 = nn.Linear(input_dim, interim_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(interim_dim, output_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)

        return x