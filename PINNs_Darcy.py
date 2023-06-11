import torch
import torch.nn as nn
import torch.optim as optim


class Darcy_eigen_nn(nn.Module):
    """
    Class to define the PINN architecture. The network consists of 6 linear layers with GELU activation.
    """

    def __init__(self, input_size=2, hidden_size=64, output_size=1):
        super(LidDrivenCavityPINN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.fc6(x)
        return x
