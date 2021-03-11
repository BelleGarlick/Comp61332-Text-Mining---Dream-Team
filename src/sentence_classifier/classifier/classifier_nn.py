import torch.nn as nn
from torch import sigmoid
from torch.tensor import Tensor

from typing import Optional


class ClassifierNN(nn.Module):
    def __init__(self, input_dim: Optional[int] = None):
        super(ClassifierNN, self).__init__()

        # TODO: confirm input dimension (sentence representation size) and output dimension (# uniq. labels)
        self.input_dim = input_dim if input_dim is not None else 120
        self.output_dim = 50

        self.fc1 = nn.Linear(self.input_dim, 160)
        self.fc2 = nn.Linear(160, 80)
        self.fc3 = nn.Linear(80, self.output_dim)

    def forward(self, x: Tensor):
        x = self.fc1(x)
        x = sigmoid(x)
        x = self.fc2(x)
        x = sigmoid(x)
        x = self.fc3(x)
        x = nn.functional.log_softmax(x, dim=1)

        return x

