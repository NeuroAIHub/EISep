from __future__ import annotations

from calendar import c
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class SimpleRNNCell(nn.Module):
    """
    A simple RNN cell implementation.

    Args:
        input_size: The number of input features.
        hidden_size: The number of features in the hidden state.
    """

    def __init__(self, input_size, hidden_size: int, output_size: int, extra_mlp: bool = False) -> None:
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
        self.Wrec_free = nn.Parameter(torch.randn((hidden_size, hidden_size)) * std)
        self.bias = nn.Parameter(torch.zeros((hidden_size,)))
        self.act = nn.Tanh()
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
        h = self.act( self.input_layer(input) + hidden@self.Wrec_free.T + self.bias )
        out = self.output_layer(h)
        return out, h

    def get_actual_weight(self):
        return self.Wrec_free

    def reset_hidden_on_done(self, hidden, done_mask):
        assert done_mask.dim() == 2 and done_mask.shape[1] == 1
        return hidden * (1 - done_mask)

    def detach_hidden(self, hidden):
        return hidden.detach()

class EIRNNCell(nn.Module):
    """
    An Excitatory-Inhibitory RNN cell implementation.

    Args:
        input_size: The number of input features.
        hidden_size: The number of features in the hidden state.
    """

    def __init__(self, input_size, hidden_size: int, output_size: int, extra_mlp: bool = False, rec_filter: str = 'relu',
                 init_choice = None) -> None:
        """Initializes the weights and biases of the RNN cell."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if rec_filter == 'relu':
            self.rec_filter   = nn.ReLU()
        elif rec_filter == 'abs':
            print('Use abs for rec filter')
            self.rec_filter   = torch.abs
        std = 1/ np.sqrt(hidden_size)
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.Wrec_free = nn.Parameter(self.rec_filter(torch.randn(hidden_size, hidden_size)*std))
        self.bias = nn.Parameter(torch.randn(hidden_size)*std)
        self.eimask_logit = nn.Parameter(torch.randn(hidden_size, 2)*std)

        self.extra_mlp = extra_mlp
        self.act = nn.Tanh()
        if extra_mlp:
            print("Using extra mlp layer in EIRNNCell")
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
        # self.output_layer = EI_Linear(hidden_size, output_size)
        if init_choice == 'ColEI':
            print('Using ColEI to initialize')
            self.init_weights_ColEI()
        elif init_choice == 'DBPN':
            print('Using DBPN to initialize')
            self.init_weights_DBPN()
        self.training = True

    def init_weights_ColEI(self):
        """
        todo
        """
        self.zero_diag = True
        self.ablate_ii = False
        self.spectral_radius = 1.5
        ei_ratio = 9
        ne = self.hidden_size * ei_ratio // (1 + ei_ratio)
        ni = self.hidden_size - ne

        self.denom = ((2 * np.pi - 1) / (2 * np.pi)) * (ne + (ne ** 2 / ni))

        sigma_we = np.sqrt(1 / self.denom)
        sigma_wi = (ne / ni) * sigma_we

        We_np = np.random.exponential(scale=sigma_we, size=(self.hidden_size, ne))
        Wi_np = -np.random.exponential(scale=sigma_wi, size=(self.hidden_size, ni))

        W = np.concatenate([We_np, Wi_np], axis=1)
        if self.zero_diag: np.fill_diagonal(W, 0)
        if self.ablate_ii: W[-ni:,-ni:] = 0
        if self.spectral_radius is not None:
            w, v = np.linalg.eig(W)
            # rho = np.max(np.real(w))
            rho = np.max( np.sqrt(np.power(np.real(w), 2) + np.power(np.imag(w), 2)) )
            W *= (self.spectral_radius / rho)

        # layer.W_pos.data = torch.from_numpy(np.absolute(W)).float()
        self.Wrec_free = nn.Parameter(torch.from_numpy(np.absolute(W)).float())
        eimask_logit = torch.zeros(self.hidden_size, 2)
        eimask_logit[:ne, 0] = 1.0  # excitatory
        eimask_logit[ne:, 1] = 1.0  # inhibitory
        self.eimask_logit = nn.Parameter(eimask_logit)
        # bias
        nn.init.zeros_(self.bias)

    def init_weights_DBPN(self):
        ei_ratio = 4
        self.frac_ex = ei_ratio / (1 + ei_ratio)
        weights = np.zeros((self.hidden_size, self.hidden_size))
        index_ranges = [(0,int(self.hidden_size*self.frac_ex)),(int(self.hidden_size*self.frac_ex),self.hidden_size)]
        signs = [1, -1]  # excitatory, inhibitory
        # for ii, (start, end) in enumerate(index_ranges):
        #     limit = 1 / np.sqrt(self.hidden_size) if signs[ii] == 1 else -1/np.sqrt(self.hidden_size)
        #     weights[:, start:end] = np.random.uniform(0, limit, size=(self.hidden_size, end-start))
        weights = np.random.uniform(0, 1/np.sqrt(self.hidden_size), size=(self.hidden_size, self.hidden_size))
        self.Wrec_free = nn.Parameter(torch.from_numpy(weights).float())
        eimask_logit = torch.zeros(self.hidden_size, 2)
        eimask_logit[:int(self.hidden_size*self.frac_ex), 0] = 1.0  # excitatory
        eimask_logit[int(self.hidden_size*self.frac_ex):, 1] = 1.0  # inhibitory
        self.eimask_logit = nn.Parameter(eimask_logit)
        stdv = 1/np.sqrt(self.hidden_size)
        b = torch.empty(self.hidden_size).uniform_(-stdv, stdv) + torch.empty(self.hidden_size).uniform_(-stdv, stdv)
        self.bias = nn.Parameter(b)

    def forward(self, x, hidden):
        # x: (B, D_in)
        B, _ = x.shape
        device  = x.device
        input_gate = self.input_layer(x)  # (B, H)
        if self.training:
            mask = F.gumbel_softmax(self.eimask_logit,
                        tau=0.1, hard=False, dim=-1)
            ei_code = mask[:, 0] - mask[:, 1]
        else:
            ei_code = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
        Wrec   = self.rec_filter(self.Wrec_free)        # (H, H)
        actual = Wrec @ torch.diag(ei_code)     # (H, H)
        # 3) rec input
        rec_in = hidden @ actual.T + self.bias          # (B, H)
        # 4) new hidden
        new_h    = self.act(input_gate + rec_in)                   # (B, H)
        ###TODO: multiple mlp layers
        # new_h1 = self.act(self.mlp_layer(new_h))
        output = self.output_layer(new_h)
        return output, new_h

    # def eval(self):
    #     self.training = False
    # def train(self, mode=True):
    #     self.training = mode

    def get_ei_entropy_loss(self):
        eimask_prob = F.softmax(self.eimask_logit, dim=-1)
        eimask_entropy = -torch.sum(eimask_prob * torch.log(eimask_prob + 1e-10), dim=-1)
        entropy_loss = torch.mean(eimask_entropy)  # Add entropy regularization
        return entropy_loss

    def get_actual_weight(self):
        Wrec = self.rec_filter(self.Wrec_free)
        eicode = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
        actual = Wrec @ torch.diag(eicode)
        return actual

    def reset_hidden_on_done(self, hidden, done_mask):
        assert done_mask.dim() == 2 and done_mask.shape[1] == 1
        return hidden * (1 - done_mask)

    def detach_hidden(self, hidden):
        return hidden.detach()


class Fixed_EIRNNCell(nn.Module):
    """
    An Excitatory-Inhibitory RNN cell implementation.

    Args:
        input_size: The number of input features.
        hidden_size: The number of features in the hidden state.
    """

    def __init__(self, input_size, hidden_size: int, output_size: int) -> None:
        """Initializes the weights and biases of the RNN cell."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rec_filter   = nn.ReLU()
        std = 1/ np.sqrt(hidden_size)
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.Wrec_free = nn.Parameter(nn.ReLU()(torch.randn(hidden_size, hidden_size)*std))
        self.bias = nn.Parameter(torch.randn(hidden_size)*std)
        self.eimask_logit = nn.Parameter(torch.randn(hidden_size, 2)*std)
        self.eimask_logit.requires_grad = False # fixed sign


        self.output_layer = nn.Linear(hidden_size, output_size)
        # self.output_layer = EI_Linear(hidden_size, output_size)
        self.act          = nn.Tanh()
        self.training = True
        for m in [self.input_layer, self.output_layer]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.normal_(m.bias, mean=0, std=std)

    def forward(self, x, hidden):
        # x: (B, D_in)
        B, _ = x.shape
        device  = x.device
        input_gate = self.input_layer(x)  # (B, H)
        ei_code = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
        Wrec   = self.rec_filter(self.Wrec_free)        # (H, H)
        actual = Wrec @ torch.diag(ei_code)     # (H, H)
        # 3) rec input
        rec_in = hidden @ actual.T + self.bias          # (B, H)
        # 4) new hidden
        new_h    = self.act(input_gate + rec_in)                   # (B, H)
        output = self.output_layer(new_h)
        return output, new_h


    def get_ei_entropy_loss(self):
        eimask_prob = F.softmax(self.eimask_logit, dim=-1)
        eimask_entropy = -torch.sum(eimask_prob * torch.log(eimask_prob + 1e-10), dim=-1)
        entropy_loss = torch.mean(eimask_entropy)  # Add entropy regularization
        return entropy_loss

    def get_actual_weight(self):
        Wrec = self.rec_filter(self.Wrec_free)
        eicode = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
        actual = Wrec @ torch.diag(eicode)
        return actual

    def reset_hidden_on_done(self, hidden, done_mask):
        assert done_mask.dim() == 2 and done_mask.shape[1] == 1
        return hidden * (1 - done_mask)

    def detach_hidden(self, hidden):
        return hidden.detach()



class EntryEIRNNCell(nn.Module):
    """
    An Excitatory-Inhibitory RNN cell implementation.

    Args:
        input_size: The number of input features.
        hidden_size: The number of features in the hidden state.
    """

    def __init__(self, input_size, hidden_size: int, output_size: int) -> None:
        """Initializes the weights and biases of the RNN cell."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rec_filter   = nn.ReLU()
        std = 1/ np.sqrt(hidden_size)
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.Wrec_free = nn.Parameter(nn.ReLU()(torch.randn(hidden_size, hidden_size)*std))
        self.bias = nn.Parameter(torch.randn(hidden_size)*std)
        self.eimask_logit = nn.Parameter(torch.randn(hidden_size, hidden_size, 2)*std)
        self.eimask_logit.requires_grad = True
        self.output_layer = nn.Linear(hidden_size, output_size)
        # self.output_layer = EI_Linear(hidden_size, output_size)
        self.act          = nn.Tanh()
        self.training = True
        for m in [self.input_layer, self.output_layer]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.normal_(m.bias, mean=0, std=std)

    def forward(self, x, hidden):
        # x: (B, D_in)
        B, _ = x.shape
        device  = x.device
        input_gate = self.input_layer(x)  # (B, H)
        if self.training:
            mask = F.gumbel_softmax(self.eimask_logit,
                        tau=0.1, hard=False, dim=-1)
            ei_code = mask[..., 0] - mask[..., 1]
        else:
            ei_code = torch.where(self.eimask_logit[..., 0] > self.eimask_logit[..., 1], 1., -1.)
        Wrec   = self.rec_filter(self.Wrec_free)        # (H, H)
        actual = Wrec * ei_code
        # 3) rec input
        rec_in = hidden @ actual.T + self.bias          # (B, H)
        # 4) new hidden
        new_h    = self.act(input_gate + rec_in)                   # (B, H)
        output = self.output_layer(new_h)
        return output, new_h


    def get_ei_entropy_loss(self):
        eimask_prob = F.softmax(self.eimask_logit, dim=-1)
        eimask_entropy = -(eimask_prob * torch.log(eimask_prob + 1e-10)).sum(-1).mean()
        entropy_loss = torch.mean(eimask_entropy)  # Add entropy regularization
        return entropy_loss

    def get_actual_weight(self):
        Wrec = self.rec_filter(self.Wrec_free)
        eicode = torch.where(self.eimask_logit[..., 0] > self.eimask_logit[..., 1], 1., -1.)
        actual = Wrec * eicode
        return actual

    def reset_hidden_on_done(self, hidden, done_mask):
        assert done_mask.dim() == 2 and done_mask.shape[1] == 1
        return hidden * (1 - done_mask)

    def detach_hidden(self, hidden):
        return hidden.detach()




class Fixed_EntryEIRNNCell(nn.Module):
    """
    An Excitatory-Inhibitory RNN cell implementation.

    Args:
        input_size: The number of input features.
        hidden_size: The number of features in the hidden state.
    """

    def __init__(self, input_size, hidden_size: int, output_size: int) -> None:
        """Initializes the weights and biases of the RNN cell."""
        print("Using Fixed_EntryEIRNNCell")
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rec_filter   = nn.ReLU()
        std = 1/ np.sqrt(hidden_size)
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.Wrec_free = nn.Parameter(nn.ReLU()(torch.randn(hidden_size, hidden_size)*std))
        self.bias = nn.Parameter(torch.randn(hidden_size)*std)
        self.eimask_logit = nn.Parameter(torch.randn(hidden_size, hidden_size, 2)*std)
        self.eimask_logit.requires_grad = False
        self.output_layer = nn.Linear(hidden_size, output_size)
        # self.output_layer = EI_Linear(hidden_size, output_size)
        self.act          = nn.Tanh()
        self.training = True
        for m in [self.input_layer, self.output_layer]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.normal_(m.bias, mean=0, std=std)

    def forward(self, x, hidden):
        # x: (B, D_in)
        B, _ = x.shape
        device  = x.device
        input_gate = self.input_layer(x)  # (B, H)
        ei_code = torch.where(self.eimask_logit[..., 0] > self.eimask_logit[..., 1], 1., -1.)
        Wrec   = self.rec_filter(self.Wrec_free)        # (H, H)
        actual = Wrec * ei_code
        # 3) rec input
        rec_in = hidden @ actual.T + self.bias          # (B, H)
        # 4) new hidden
        new_h    = self.act(input_gate + rec_in)                   # (B, H)
        output = self.output_layer(new_h)
        return output, new_h

    def get_actual_weight(self):
        Wrec = self.rec_filter(self.Wrec_free)
        eicode = torch.where(self.eimask_logit[..., 0] > self.eimask_logit[..., 1], 1., -1.)
        actual = Wrec * eicode
        return actual

    def reset_hidden_on_done(self, hidden, done_mask):
        assert done_mask.dim() == 2 and done_mask.shape[1] == 1
        return hidden * (1 - done_mask)

    def detach_hidden(self, hidden):
        return hidden.detach()


class SimpleGRUCell(nn.Module):
    """
    A simple GRU cell implementation.

    Args:
        input_size: The number of input features.
        hidden_size: The number of features in the hidden state.
    """

    def __init__(self, input_size, hidden_size: int, output_size: int) -> None:
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
        self.input_layer = nn.Linear(input_size, hidden_size*3)
        std = 1.0 / np.sqrt(hidden_size)
        self.Wrec_free = nn.Parameter(torch.randn((hidden_size*3, hidden_size)) * std)
        self.bias = nn.Parameter(torch.zeros((hidden_size*3,)))
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.act = nn.Tanh()

        # init input_layer orthogonally
        for m in [self.input_layer, self.output_layer]:
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
        input_gate = self.input_layer(input)  # (B, H*3)
        input_h, input_z, input_r = torch.chunk(input_gate, dim=-1, chunks=3)
        bias_h, bias_z, bias_r = torch.chunk(self.bias, dim=-1, chunks=3)
        Wrec_h, Wrec_z, Wrec_r = torch.chunk(self.Wrec_free, dim=0, chunks=3)
        z = torch.sigmoid(input_z + hidden @ Wrec_z.T + bias_z)
        r = torch.sigmoid(input_r + hidden @ Wrec_r.T + bias_r)
        h_tilde = self.act(input_h + (r * hidden) @ Wrec_h.T + bias_h)
        new_h = (1 - z) * hidden + z * h_tilde
        output = self.output_layer(new_h)
        return output, new_h


    def get_actual_weight(self):
        return self.Wrec_free

class EIGRUCell(nn.Module):
    """
    An Excitatory-Inhibitory GRU cell implementation.

    Args:
        input_size: The number of input features.
        hidden_size: The number of features in the hidden state.
    """

    def __init__(self, input_size, hidden_size: int, output_size: int) -> None:
        """Initializes the weights and biases of the RNN cell."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rec_filter   = nn.ReLU()
        std = 1/ np.sqrt(hidden_size)
        self.input_layer = nn.Linear(input_size, hidden_size*3)
        self.Wrec_free = nn.Parameter(nn.ReLU()(torch.randn(hidden_size*3, hidden_size)*std))
        self.bias = nn.Parameter(torch.randn(hidden_size*3)*std)
        self.eimask_logit = nn.Parameter(torch.randn(hidden_size, 2)*std)

        self.output_layer = nn.Linear(hidden_size, output_size)
        # self.output_layer = EI_Linear(hidden_size, output_size)
        self.act          = nn.Tanh()
        self.training = True
        for m in [self.input_layer, self.output_layer]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.normal_(m.bias, mean=0, std=std)

    def forward(self, x, hidden):
        # x: (B, D_in)
        B, _ = x.shape
        device  = x.device
        input_gate = self.input_layer(x)  # (B, H*3)
        input_h, input_z, input_r = torch.chunk(input_gate, dim=-1, chunks=3)
        bias_h, bias_z, bias_r = torch.chunk(self.bias, dim=-1, chunks=3)
        if self.training:
            mask = F.gumbel_softmax(self.eimask_logit,
                        tau=0.1, hard=False, dim=-1)
            ei_code = mask[:, 0] - mask[:, 1]
        else:
            ei_code = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
        Wrec   = self.rec_filter(self.Wrec_free)        # (H*3, H)
        actual = Wrec @ torch.diag(ei_code)     # (H*3, H)
        Wrec_h, Wrec_z, Wrec_r = torch.chunk(actual, dim=0, chunks=3)
        z = torch.sigmoid(input_z + hidden @ Wrec_z.T + bias_z)
        r = torch.sigmoid(input_r + hidden @ Wrec_r.T + bias_r)
        h_tilde = self.act(input_h + (r * hidden) @ Wrec_h.T + bias_h)
        new_h = (1 - z) * hidden + z * h_tilde
        output = self.output_layer(new_h)

        return output, new_h


    def get_ei_entropy_loss(self):
        eimask_prob = F.softmax(self.eimask_logit, dim=-1)
        eimask_entropy = -torch.sum(eimask_prob * torch.log(eimask_prob + 1e-10), dim=-1)
        entropy_loss = torch.mean(eimask_entropy)  # Add entropy regularization
        return entropy_loss

    def get_actual_weight(self):
        Wrec = self.rec_filter(self.Wrec_free)
        eicode = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
        actual = Wrec @ torch.diag(eicode)
        return actual


class Noshare_EIGRUCell(nn.Module):
    """
    An Excitatory-Inhibitory GRU cell implementation.

    Args:
        input_size: The number of input features.
        hidden_size: The number of features in the hidden state.
    """

    def __init__(self, input_size, hidden_size: int, output_size: int) -> None:
        """Initializes the weights and biases of the RNN cell."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rec_filter   = nn.ReLU()
        std = 1/ np.sqrt(hidden_size)
        self.input_layer = nn.Linear(input_size, hidden_size*3)
        self.Wrec_free = nn.Parameter(nn.ReLU()(torch.randn(hidden_size*3, hidden_size)*std))
        self.bias = nn.Parameter(torch.randn(hidden_size*3)*std)
        self.eimask_logit = nn.Parameter(torch.randn(hidden_size*3, 2)*std)

        self.output_layer = nn.Linear(hidden_size, output_size)
        # self.output_layer = EI_Linear(hidden_size, output_size)
        self.act          = nn.Tanh()
        self.training = True
        for m in [self.input_layer, self.output_layer]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.normal_(m.bias, mean=0, std=std)

    def forward(self, x, hidden):
        # x: (B, D_in)
        B, _ = x.shape
        device  = x.device
        input_gate = self.input_layer(x)  # (B, H*3)
        input_h, input_z, input_r = torch.chunk(input_gate, dim=-1, chunks=3)
        bias_h, bias_z, bias_r = torch.chunk(self.bias, dim=-1, chunks=3)
        if self.training:
            mask = F.gumbel_softmax(self.eimask_logit,
                        tau=0.1, hard=False, dim=-1)
            ei_code = mask[:, 0] - mask[:, 1]
        else:
            ei_code = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
        ei_code_h, ei_code_z, ei_code_r = torch.chunk(ei_code, dim=0, chunks=3)
        # Wrec   = self.rec_filter(self.Wrec_free)        # (H*3, H)
        # actual = Wrec @ torch.diag(ei_code)     # (H*3, H)
        # Wrec_h, Wrec_z, Wrec_r = torch.chunk(actual, dim=0, chunks=3)
        Wrec_h, Wrec_z, Wrec_r = torch.chunk(self.Wrec_free, dim=0, chunks=3)
        Wrec_h = self.rec_filter(Wrec_h) @ torch.diag(ei_code_h)
        Wrec_z = self.rec_filter(Wrec_z) @ torch.diag(ei_code_z)
        Wrec_r = self.rec_filter(Wrec_r) @ torch.diag(ei_code_r)
        z = torch.sigmoid(input_z + hidden @ Wrec_z.T + bias_z)
        r = torch.sigmoid(input_r + hidden @ Wrec_r.T + bias_r)
        h_tilde = self.act(input_h + (r * hidden) @ Wrec_h.T + bias_h)
        new_h = (1 - z) * hidden + z * h_tilde
        output = self.output_layer(new_h)

        return output, new_h


    def get_ei_entropy_loss(self):
        eimask_prob = F.softmax(self.eimask_logit, dim=-1)
        eimask_entropy = -torch.sum(eimask_prob * torch.log(eimask_prob + 1e-10), dim=-1)
        entropy_loss = torch.mean(eimask_entropy)  # Add entropy regularization
        return entropy_loss

    def get_actual_weight(self):
        Wrec = self.rec_filter(self.Wrec_free)
        eicode = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
        actual = Wrec @ torch.diag(eicode)
        return actual


class SimpleLSTMCell(nn.Module):
    """
    A simple GRU cell implementation.

    Args:
        input_size: The number of input features.
        hidden_size: The number of features in the hidden state.
    """

    def __init__(self, input_size, hidden_size: int, output_size: int) -> None:
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
        self.input_layer = nn.Linear(input_size, hidden_size*4)
        std = 1.0 / np.sqrt(hidden_size)
        self.Wrec_free = nn.Parameter(torch.randn((hidden_size*4, hidden_size)) * std)
        self.bias = nn.Parameter(torch.zeros((hidden_size*4,)))
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.act = nn.Tanh()

        # init input_layer orthogonally
        for m in [self.input_layer, self.output_layer]:
            nn.init.orthogonal_(m.weight)
            nn.init.normal_(m.bias, mean=0, std=std)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM cell.

        Args:
            input: The input tensor with shape (batch_size, input_size).
            hidden: The hidden state tensor with shape (batch_size, hidden_size).

        Returns:
            The next hidden state tensor with shape (batch_size, hidden_size).
        # """
        h, c = hidden
        input_gate = self.input_layer(input)  # (B, H*4)
        input_i, input_f, input_g, input_o = torch.chunk(input_gate, dim=-1, chunks=4)
        bias_i, bias_f, bias_g, bias_o = torch.chunk(self.bias, dim=-1, chunks=4)
        h_gate = h @ self.Wrec_free.T  # (B, H*4)
        h_i, h_f, h_g, h_o = torch.chunk(h_gate, dim=-1, chunks=4)
        i = torch.sigmoid(input_i + h_i + bias_i)
        f = torch.sigmoid(input_f + h_f + bias_f)
        g = torch.tanh(input_g + h_g + bias_g)
        o = torch.sigmoid(input_o + h_o + bias_o)
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        output = self.output_layer(new_h)
        return output, (new_h, new_c)

    def init_hidden(self, batch_size: int):
        """Initializes the hidden state tensor."""
        h = torch.zeros(batch_size, self.hidden_size).to(self.Wrec_free.device)
        c = torch.zeros(batch_size, self.hidden_size).to(self.Wrec_free.device)
        return (h, c)

    def detach_hidden(self, hidden):
        h, c = hidden
        return (h.detach(), c.detach())

    def reset_hidden_on_done(self, hidden, done_mask):
        assert done_mask.dim() == 2 and done_mask.shape[1] == 1
        h, c = hidden
        h = h * (1 - done_mask)
        c = c * (1 - done_mask)
        return (h, c)

    def get_actual_weight(self):
        return self.Wrec_free


class EILSTMCell(nn.Module):
    """
    An Excitatory-Inhibitory GRU cell implementation.

    Args:
        input_size: The number of input features.
        hidden_size: The number of features in the hidden state.
    """

    def __init__(self, input_size, hidden_size: int, output_size: int) -> None:
        """Initializes the weights and biases of the RNN cell."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rec_filter   = nn.ReLU()
        std = 1/ np.sqrt(hidden_size)
        self.input_layer = nn.Linear(input_size, hidden_size*4)
        self.Wrec_free = nn.Parameter(nn.ReLU()(torch.randn(hidden_size*4, hidden_size)*std))
        self.bias = nn.Parameter(torch.randn(hidden_size*4)*std)
        self.eimask_logit = nn.Parameter(torch.randn(hidden_size, 2)*std)

        self.output_layer = nn.Linear(hidden_size, output_size)
        # self.output_layer = EI_Linear(hidden_size, output_size)
        self.act          = nn.Tanh()
        self.training = True
        for m in [self.input_layer, self.output_layer]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.normal_(m.bias, mean=0, std=std)

    def forward(self, x, hidden):
        # x: (B, D_in)
        B, _ = x.shape
        device  = x.device
        h, c = hidden
        input_gate = self.input_layer(x)  # (B, H*4)
        input_i, input_f, input_g, input_o = torch.chunk(input_gate, dim=-1, chunks=4)
        bias_i, bias_f, bias_g, bias_o = torch.chunk(self.bias, dim=-1, chunks=4)
        if self.training:
            mask = F.gumbel_softmax(self.eimask_logit,
                        tau=0.1, hard=False, dim=-1)
            ei_code = mask[:, 0] - mask[:, 1]
        else:
            ei_code = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
        Wrec   = self.rec_filter(self.Wrec_free)        # (H*4, H)
        actual = Wrec @ torch.diag(ei_code)     # (H*4, H)
        h_gate = h @ actual.T  # (B, H*4)
        h_i, h_f, h_g, h_o = torch.chunk(h_gate, dim=-1, chunks=4)
        i = torch.sigmoid(input_i + h_i + bias_i)
        f = torch.sigmoid(input_f + h_f + bias_f)
        g = torch.tanh(input_g + h_g + bias_g)
        o = torch.sigmoid(input_o + h_o + bias_o)
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        output = self.output_layer(new_h)

        return output, (new_h, new_c)

    def init_hidden(self, batch_size: int):
        """Initializes the hidden state tensor."""
        h = torch.zeros(batch_size, self.hidden_size).to(self.Wrec_free.device)
        c = torch.zeros(batch_size, self.hidden_size).to(self.Wrec_free.device)
        return (h, c)

    def detach_hidden(self, hidden):
        h, c = hidden
        return (h.detach(), c.detach())

    def reset_hidden_on_done(self, hidden, done_mask):
        assert done_mask.dim() == 2 and done_mask.shape[1] == 1
        h, c = hidden
        h = h * (1 - done_mask)
        c = c * (1 - done_mask)
        return (h, c)

    def get_ei_entropy_loss(self):
        eimask_prob = F.softmax(self.eimask_logit, dim=-1)
        eimask_entropy = -torch.sum(eimask_prob * torch.log(eimask_prob + 1e-10), dim=-1)
        entropy_loss = torch.mean(eimask_entropy)  # Add entropy regularization
        return entropy_loss

    def get_actual_weight(self):
        Wrec = self.rec_filter(self.Wrec_free)
        eicode = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
        actual = Wrec @ torch.diag(eicode)
        return actual


class Noshare_EILSTMCell(nn.Module):
    """
    An Excitatory-Inhibitory GRU cell implementation.

    Args:
        input_size: The number of input features.
        hidden_size: The number of features in the hidden state.
    """

    def __init__(self, input_size, hidden_size: int, output_size: int) -> None:
        """Initializes the weights and biases of the RNN cell."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rec_filter   = nn.ReLU()
        std = 1/ np.sqrt(hidden_size)
        self.input_layer = nn.Linear(input_size, hidden_size*4)
        self.Wrec_free = nn.Parameter(nn.ReLU()(torch.randn(hidden_size*4, hidden_size)*std))
        self.bias = nn.Parameter(torch.randn(hidden_size*4)*std)
        self.eimask_logit = nn.Parameter(torch.randn(hidden_size*4, 2)*std)

        self.output_layer = nn.Linear(hidden_size, output_size)
        # self.output_layer = EI_Linear(hidden_size, output_size)
        self.act          = nn.Tanh()
        self.training = True
        for m in [self.input_layer, self.output_layer]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.normal_(m.bias, mean=0, std=std)

    def forward(self, x, hidden):
        # x: (B, D_in)
        B, _ = x.shape
        device  = x.device
        h, c = hidden
        input_gate = self.input_layer(x)  # (B, H*4)
        input_i, input_f, input_g, input_o = torch.chunk(input_gate, dim=-1, chunks=4)
        bias_i, bias_f, bias_g, bias_o = torch.chunk(self.bias, dim=-1, chunks=4)
        if self.training:
            mask = F.gumbel_softmax(self.eimask_logit,
                        tau=0.1, hard=False, dim=-1)
            ei_code = mask[:, 0] - mask[:, 1]
        else:
            ei_code = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
        eicode_i, eicode_f, eicode_g, eicode_o = torch.chunk(ei_code, dim=0, chunks=4)

        Wrec   = self.rec_filter(self.Wrec_free)        # (H*4, H)
        Wrec_i, Wrec_f, Wrec_g, Wrec_o = torch.chunk(Wrec, dim=0, chunks=4)
        actual_Wi = Wrec_i @ torch.diag(eicode_i)
        actual_Wf = Wrec_f @ torch.diag(eicode_f)
        actual_Wg = Wrec_g @ torch.diag(eicode_g)
        actual_Wo = Wrec_o @ torch.diag(eicode_o)

        h_i = h @ actual_Wi.T
        h_f = h @ actual_Wf.T
        h_g = h @ actual_Wg.T
        h_o = h @ actual_Wo.T

        i = torch.sigmoid(input_i + h_i + bias_i)
        f = torch.sigmoid(input_f + h_f + bias_f)
        g = torch.tanh(input_g + h_g + bias_g)
        o = torch.sigmoid(input_o + h_o + bias_o)
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        output = self.output_layer(new_h)

        return output, (new_h, new_c)

    def init_hidden(self, batch_size: int):
        """Initializes the hidden state tensor."""
        h = torch.zeros(batch_size, self.hidden_size).to(self.Wrec_free.device)
        c = torch.zeros(batch_size, self.hidden_size).to(self.Wrec_free.device)
        return (h, c)

    def detach_hidden(self, hidden):
        h, c = hidden
        return (h.detach(), c.detach())

    def reset_hidden_on_done(self, hidden, done_mask):
        assert done_mask.dim() == 2 and done_mask.shape[1] == 1
        h, c = hidden
        h = h * (1 - done_mask)
        c = c * (1 - done_mask)
        return (h, c)

    def get_ei_entropy_loss(self):
        eimask_prob = F.softmax(self.eimask_logit, dim=-1)
        eimask_entropy = -torch.sum(eimask_prob * torch.log(eimask_prob + 1e-10), dim=-1)
        entropy_loss = torch.mean(eimask_entropy)  # Add entropy regularization
        return entropy_loss

    def get_actual_weight(self):
        Wrec = self.rec_filter(self.Wrec_free)
        eicode = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
        actual = Wrec @ torch.diag(eicode)
        return actual



class Fixed_EILSTMCell(nn.Module):
    """
    An Excitatory-Inhibitory GRU cell implementation.

    Args:
        input_size: The number of input features.
        hidden_size: The number of features in the hidden state.
    """

    def __init__(self, input_size, hidden_size: int, output_size: int) -> None:
        """Initializes the weights and biases of the RNN cell."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rec_filter   = nn.ReLU()
        std = 1/ np.sqrt(hidden_size)
        self.input_layer = nn.Linear(input_size, hidden_size*4)
        self.Wrec_free = nn.Parameter(nn.ReLU()(torch.randn(hidden_size*4, hidden_size)*std))
        self.bias = nn.Parameter(torch.randn(hidden_size*4)*std)
        self.eimask_logit = nn.Parameter(torch.randn(hidden_size, 2)*std)
        self.eimask_logit.requires_grad = False

        self.output_layer = nn.Linear(hidden_size, output_size)
        # self.output_layer = EI_Linear(hidden_size, output_size)
        self.act          = nn.Tanh()
        self.training = True
        for m in [self.input_layer, self.output_layer]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.normal_(m.bias, mean=0, std=std)

    def forward(self, x, hidden):
        # x: (B, D_in)
        B, _ = x.shape
        device  = x.device
        h, c = hidden
        input_gate = self.input_layer(x)  # (B, H*4)
        input_i, input_f, input_g, input_o = torch.chunk(input_gate, dim=-1, chunks=4)
        bias_i, bias_f, bias_g, bias_o = torch.chunk(self.bias, dim=-1, chunks=4)
        ei_code = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
        Wrec   = self.rec_filter(self.Wrec_free)        # (H*4, H)
        actual = Wrec @ torch.diag(ei_code)     # (H*4, H)
        h_gate = h @ actual.T  # (B, H*4)
        h_i, h_f, h_g, h_o = torch.chunk(h_gate, dim=-1, chunks=4)
        i = torch.sigmoid(input_i + h_i + bias_i)
        f = torch.sigmoid(input_f + h_f + bias_f)
        g = torch.tanh(input_g + h_g + bias_g)
        o = torch.sigmoid(input_o + h_o + bias_o)
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        output = self.output_layer(new_h)

        return output, (new_h, new_c)

    def init_hidden(self, batch_size: int):
        """Initializes the hidden state tensor."""
        h = torch.zeros(batch_size, self.hidden_size).to(self.Wrec_free.device)
        c = torch.zeros(batch_size, self.hidden_size).to(self.Wrec_free.device)
        return (h, c)

    def detach_hidden(self, hidden):
        h, c = hidden
        return (h.detach(), c.detach())

    def reset_hidden_on_done(self, hidden, done_mask):
        assert done_mask.dim() == 2 and done_mask.shape[1] == 1
        h, c = hidden
        h = h * (1 - done_mask)
        c = c * (1 - done_mask)
        return (h, c)

    def get_ei_entropy_loss(self):
        eimask_prob = F.softmax(self.eimask_logit, dim=-1)
        eimask_entropy = -torch.sum(eimask_prob * torch.log(eimask_prob + 1e-10), dim=-1)
        entropy_loss = torch.mean(eimask_entropy)  # Add entropy regularization
        return entropy_loss

    def get_actual_weight(self):
        Wrec = self.rec_filter(self.Wrec_free)
        eicode = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
        actual = Wrec @ torch.diag(eicode)
        return actual



class EntryEILSTMCell(nn.Module):
    """
    An Excitatory-Inhibitory GRU cell implementation.

    Args:
        input_size: The number of input features.
        hidden_size: The number of features in the hidden state.
    """

    def __init__(self, input_size, hidden_size: int, output_size: int) -> None:
        """Initializes the weights and biases of the RNN cell."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rec_filter   = nn.ReLU()
        std = 1/ np.sqrt(hidden_size)
        self.input_layer = nn.Linear(input_size, hidden_size*4)
        self.Wrec_free = nn.Parameter(nn.ReLU()(torch.randn(hidden_size*4, hidden_size)*std))
        self.bias = nn.Parameter(torch.randn(hidden_size*4)*std)
        self.eimask_logit = nn.Parameter(torch.randn(hidden_size, hidden_size, 2)*std)
        self.eimask_logit.requires_grad = True

        self.output_layer = nn.Linear(hidden_size, output_size)
        # self.output_layer = EI_Linear(hidden_size, output_size)
        self.act          = nn.Tanh()
        self.training = True
        for m in [self.input_layer, self.output_layer]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.normal_(m.bias, mean=0, std=std)

    def forward(self, x, hidden):
        # x: (B, D_in)
        B, _ = x.shape
        device  = x.device
        h, c = hidden
        input_gate = self.input_layer(x)  # (B, H*4)
        input_i, input_f, input_g, input_o = torch.chunk(input_gate, dim=-1, chunks=4)
        bias_i, bias_f, bias_g, bias_o = torch.chunk(self.bias, dim=-1, chunks=4)
        if self.training:
            mask = F.gumbel_softmax(self.eimask_logit,
                        tau=0.1, hard=False, dim=-1)
            ei_code = mask[..., 0] - mask[..., 1]
        else:
            ei_code = torch.where(self.eimask_logit[..., 0] > self.eimask_logit[..., 1], 1., -1.)
        eicode_repeat = ei_code.repeat(4,1,1).reshape(4*h.shape[-1], h.shape[-1])
        Wrec   = self.rec_filter(self.Wrec_free)        # (H*4, H)
        actual = Wrec * eicode_repeat     # (H*4, H)
        h_gate = h @ actual.T  # (B, H*4)
        h_i, h_f, h_g, h_o = torch.chunk(h_gate, dim=-1, chunks=4)
        i = torch.sigmoid(input_i + h_i + bias_i)
        f = torch.sigmoid(input_f + h_f + bias_f)
        g = torch.tanh(input_g + h_g + bias_g)
        o = torch.sigmoid(input_o + h_o + bias_o)
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        output = self.output_layer(new_h)

        return output, (new_h, new_c)

    def init_hidden(self, batch_size: int):
        """Initializes the hidden state tensor."""
        h = torch.zeros(batch_size, self.hidden_size).to(self.Wrec_free.device)
        c = torch.zeros(batch_size, self.hidden_size).to(self.Wrec_free.device)
        return (h, c)

    def detach_hidden(self, hidden):
        h, c = hidden
        return (h.detach(), c.detach())

    def reset_hidden_on_done(self, hidden, done_mask):
        assert done_mask.dim() == 2 and done_mask.shape[1] == 1
        h, c = hidden
        h = h * (1 - done_mask)
        c = c * (1 - done_mask)
        return (h, c)

    def get_ei_entropy_loss(self):
        eimask_prob = F.softmax(self.eimask_logit, dim=-1)
        eimask_entropy = -torch.sum(eimask_prob * torch.log(eimask_prob + 1e-10), dim=-1)
        entropy_loss = torch.mean(eimask_entropy)  # Add entropy regularization
        return entropy_loss

    def get_actual_weight(self):
        Wrec = self.rec_filter(self.Wrec_free)
        ei_code = torch.where(self.eimask_logit[..., 0] > self.eimask_logit[..., 1], 1., -1.)
        eicode_repeat = ei_code.repeat(4,1,1).reshape(4*h.shape[-1], h.shape[-1])
        actual = Wrec * eicode_repeat     # (H*4, H)
        return actual


class Fixed_EntryEILSTMCell(nn.Module):
    """
    An Excitatory-Inhibitory GRU cell implementation.

    Args:
        input_size: The number of input features.
        hidden_size: The number of features in the hidden state.
    """

    def __init__(self, input_size, hidden_size: int, output_size: int) -> None:
        """Initializes the weights and biases of the RNN cell."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rec_filter   = nn.ReLU()
        std = 1/ np.sqrt(hidden_size)
        self.input_layer = nn.Linear(input_size, hidden_size*4)
        self.Wrec_free = nn.Parameter(nn.ReLU()(torch.randn(hidden_size*4, hidden_size)*std))
        self.bias = nn.Parameter(torch.randn(hidden_size*4)*std)
        self.eimask_logit = nn.Parameter(torch.randn(hidden_size, hidden_size, 2)*std)
        self.eimask_logit.requires_grad = False

        self.output_layer = nn.Linear(hidden_size, output_size)
        # self.output_layer = EI_Linear(hidden_size, output_size)
        self.act          = nn.Tanh()
        self.training = True
        for m in [self.input_layer, self.output_layer]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.normal_(m.bias, mean=0, std=std)

    def forward(self, x, hidden):
        # x: (B, D_in)
        B, _ = x.shape
        device  = x.device
        h, c = hidden
        input_gate = self.input_layer(x)  # (B, H*4)
        input_i, input_f, input_g, input_o = torch.chunk(input_gate, dim=-1, chunks=4)
        bias_i, bias_f, bias_g, bias_o = torch.chunk(self.bias, dim=-1, chunks=4)
        ei_code = torch.where(self.eimask_logit[..., 0] > self.eimask_logit[..., 1], 1., -1.)
        eicode_repeat = ei_code.repeat(4,1,1).reshape(4*h.shape[-1], h.shape[-1])
        Wrec   = self.rec_filter(self.Wrec_free)        # (H*4, H)
        actual = Wrec * eicode_repeat     # (H*4, H)
        h_gate = h @ actual.T  # (B, H*4)
        h_i, h_f, h_g, h_o = torch.chunk(h_gate, dim=-1, chunks=4)
        i = torch.sigmoid(input_i + h_i + bias_i)
        f = torch.sigmoid(input_f + h_f + bias_f)
        g = torch.tanh(input_g + h_g + bias_g)
        o = torch.sigmoid(input_o + h_o + bias_o)
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        output = self.output_layer(new_h)

        return output, (new_h, new_c)

    def init_hidden(self, batch_size: int):
        """Initializes the hidden state tensor."""
        h = torch.zeros(batch_size, self.hidden_size).to(self.Wrec_free.device)
        c = torch.zeros(batch_size, self.hidden_size).to(self.Wrec_free.device)
        return (h, c)

    def detach_hidden(self, hidden):
        h, c = hidden
        return (h.detach(), c.detach())

    def reset_hidden_on_done(self, hidden, done_mask):
        assert done_mask.dim() == 2 and done_mask.shape[1] == 1
        h, c = hidden
        h = h * (1 - done_mask)
        c = c * (1 - done_mask)
        return (h, c)

    def get_ei_entropy_loss(self):
        eimask_prob = F.softmax(self.eimask_logit, dim=-1)
        eimask_entropy = -torch.sum(eimask_prob * torch.log(eimask_prob + 1e-10), dim=-1)
        entropy_loss = torch.mean(eimask_entropy)  # Add entropy regularization
        return entropy_loss

    def get_actual_weight(self):
        Wrec = self.rec_filter(self.Wrec_free)
        ei_code = torch.where(self.eimask_logit[..., 0] > self.eimask_logit[..., 1], 1., -1.)
        eicode_repeat = ei_code.repeat(4,1,1).reshape(4*h.shape[-1], h.shape[-1])
        actual = Wrec * eicode_repeat     # (H*4, H)
        return actual
