import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import pickle as pkl

def activation_derivative(act_module, postact, preact=None):
    """Compute the derivative of the activation function.
    preact: (B, H)
    returns: (B, H) act'(preact).
    """
    if isinstance(act_module, torch.nn.ReLU):
        return torch.where(preact > 0, 1.0, 0.0)
    if isinstance(act_module, torch.nn.Tanh):
        return 1 - postact ** 2
    if isinstance(act_module, torch.nn.Softplus):
        return 1 - torch.exp(-postact)
    msg = f"Derivative for {type(act_module)} not implemented."
    raise NotImplementedError(msg)


class RL_Hidden_Buffer():
    def __init__(self, size, state_dim=1, action_dim=1, hidden_dim=1, n_h=1, frame_shape=None):
        self.state = np.zeros((size, state_dim))
        self.action = np.zeros((size, action_dim))
        self.hidden_state = np.zeros((size, n_h, hidden_dim))
        self.reward = np.zeros((size, 1))
        self.done = np.zeros((size, 1))
        self.ptr = 0
        self.size = size
        self.n_h = n_h
        if frame_shape is not None:
            self.frame = np.zeros((size, *frame_shape))
    def add(self, state, action, hidden_state, reward, done, frame=None):
        assert state.shape[-1] == self.state.shape[-1]
        assert action.shape[-1] == self.action.shape[-1]
        assert hidden_state[0].shape[-1] == self.hidden_state.shape[-1]
        self.state[self.ptr] = state
        self.action[self.ptr] = action.detach().cpu().numpy()
        for i in range(self.n_h):
            self.hidden_state[self.ptr, i] = hidden_state[i].detach().cpu().numpy()
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        if frame is not None:
            self.frame[self.ptr] = frame
        self.ptr = (self.ptr + 1) % self.size
        return
    def save(self, save_path):
        pkl.dump(self, open(save_path, 'wb'))
        print('ptr:', self.ptr, save_path)
    def load(self, load_path):
        buffer = pkl.load(open(load_path, 'rb'))
        self.state = buffer.state
        self.action = buffer.action
        self.hidden_state = buffer.hidden_state
        self.reward = buffer.reward
        self.done = buffer.done
        self.ptr = buffer.ptr
        self.size = buffer.size
        self.n_h = buffer.n_h
        if hasattr(buffer, 'frame'):
            self.frame = buffer.frame
        print(load_path, 'size:', self.ptr)
        return
