from __future__ import annotations

from calendar import c
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F




class FFNCell(nn.Module):
    """
    A simple RNN cell implementation.

    Args:
        input_size: The number of input features.
        hidden_size: The number of features in the hidden state.
    """

    def __init__(self, input_size, hidden_size: int, output_size: int, extra_mlp: bool = False, ei_linear:bool = False) -> None:
        """Initializes the weights and biases of the RNN cell."""
        super().__init__()
        self.input_size = input_size
        # self.h, self.w, self.c = self.input_size
        self.hidden_size = hidden_size
        # h=210, w=160, c=3 for Breakout
        # self.input_layer = nn.Sequential(nn.Conv2d(self.c, 16, kernel_size=8, stride=4), nn.ReLU(),
        #                                  nn.Conv2d(16, 32, kernel_size=8, stride=4), nn.ReLU(),
        #                                  nn.Conv2d(32, 32, kernel_size=4, stride=3), nn.ReLU(),
        #                                  nn.Flatten(),
        #                                  nn.Linear(192, hidden_size))
        self.input_layer = nn.Linear(input_size, hidden_size)
        std = 1.0 / np.sqrt(hidden_size)
        # self.Wrec_free = nn.Parameter(torch.randn((hidden_size, hidden_size)) * std)
        # self.bias = nn.Parameter(torch.zeros((hidden_size,)))
        self.act = nn.Tanh()
        self.ei_linear = ei_linear
        if ei_linear:
            print('Using EI_Linear as the middle layer')
            self.middle_layer = EI_Linear(hidden_size, hidden_size)
        else:
            print('Using Simple as the middle layer')
            self.middle_layer = nn.Linear(hidden_size, hidden_size)

        if extra_mlp:
            print("Using extra mlp layer in SimpleRNNCell")
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                self.act,
                nn.Linear(hidden_size, output_size)
            )
        else:
            self.output_layer = nn.Linear(hidden_size, output_size)


        # init input_layer orthogonally
        for m in [self.input_layer, self.output_layer]:
            if isinstance(m, nn.Sequential):
                for layer in m:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight)
                        nn.init.normal_(layer.bias, mean=0, std=std)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.normal_(m.bias, mean=0, std=std)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RNN cell.

        Args:
            input: The input tensor with shape (batch_size, input_size).
            hidden: The hidden state tensor with shape (batch_size, hidden_size).

        Returns:
            The next hidden state tensor with shape (batch_size, hidden_size).
        # """
        # assert input.shape[1:] == (self.h, self.w, self.c)
        # input_permute = input.permute(0,3,1,2)  # reshape to (batch_size, c, h, w)
        # h = self.act( self.input_layer(input) + hidden@self.Wrec_free.T + self.bias )
        # out = self.output_layer(h)
        h = self.act(self.middle_layer(self.act(self.input_layer(input))))
        out = self.output_layer(h)
        return out, h

    def get_actual_weight(self):
        return self.middle_layer.weight

    def reset_hidden_on_done(self, hidden, done_mask):
        assert done_mask.dim() == 2 and done_mask.shape[1] == 1
        return hidden * (1 - done_mask)

    def detach_hidden(self, hidden):
        return hidden.detach()

    def get_ei_entropy_loss(self):
        if self.ei_linear:
            return self.middle_layer.get_ei_entropy_loss()
        else:
            return 0.0



class EI_Linear(nn.Module):
    def __init__(self, input_size, output_size, entropy_loss=True):
        super().__init__()
        std = 1/np.sqrt(input_size)
        self.rec_filter = nn.ReLU()
        self.weight = nn.Parameter(nn.ReLU()(torch.randn((output_size, input_size))*std))
        self.bias = nn.Parameter(torch.randn(output_size)*std)
        self.eimask_logit = nn.Parameter(torch.randn(input_size, 2))
        self.entropy_loss = entropy_loss
        nn.init.orthogonal_(self.weight)
        nn.init.normal(self.bias)

    def forward(self, x):
        if self.training:
            mask = F.gumbel_softmax(self.eimask_logit,
                        tau=0.1, hard=False, dim=-1)
            # mask = F.softmax(self.eimask_logit/temperature, dim=-1)
            ei_code = mask[:, 0] - mask[:, 1]
        else:
            ei_code = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
        W   = self.rec_filter(self.weight)        # (O, I)
        actual = W @ torch.diag(ei_code)     # (O, I)
        y = x @ actual.T + self.bias          # (B, O)
        return y

    def get_ei_entropy_loss(self):
        eimask_prob = F.softmax(self.eimask_logit, dim=-1)
        eimask_entropy = -torch.sum(eimask_prob * torch.log(eimask_prob + 1e-10), dim=-1)
        entropy_loss = torch.mean(eimask_entropy)  # Add entropy regularization
        return entropy_loss
