import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class LipschitzPReLU(torch.nn.Module):

    def __init__(self, num_parameters: int = 1, init: float = 0.25,
                 device=None, dtype=None) -> None:
        self.num_parameters = num_parameters
        super(LipschitzPReLU, self).__init__()
        self.weight = nn.Parameter(torch.empty(num_parameters).fill_(init))

    def forward(self, input: Tensor) -> Tensor:
        return F.prelu(input, torch.clip(self.weight, -1, 1))

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)