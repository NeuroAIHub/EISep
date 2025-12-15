
from termios import B1152000
import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F

from dale_lib.dense_layers import DenseLayer, EiDense, ColumnEiDense
from dale_lib.song_rnn import ColumnEiCell
from dale_lib.update_policies import EiRNN_Adam_UpdatePolicy, ColumnEiAdam_Clip, EiRNN_cSGD_UpdatePolicy, DalesANN_cSGD_UpdatePolicy, ColumnEiSGD_Clip, ColumnEiDenseSGD
from dale_lib.init_policies import EiDenseWeightInit_WexMean, U_TorchInit, W_TorchInit, ColumnEiCell_W_InitPolicy, ColumnEi_FirstCell_U_InitPolicy, EiRNNCell_U_InitPolicy, EiRNNCell_W_InitPolicy, W_UniformInit
from docs.examples.net.dale_util import activation_derivative


class ColEI(nn.Module):
    def __init__(self, num_h, ob_size, act_size, optimizer='sgd'):
        super(ColEI, self).__init__()
        self.hidden_size = num_h
        self.input_size = ob_size
        self.output_size = act_size
        n_hidden = self.hidden_size
        max_gn = 10
        non_linearity = nn.Softplus()
        i2h_init_policy=ColumnEi_FirstCell_U_InitPolicy(dataset='PennTreebank')
        h2h_init_policy=ColumnEiCell_W_InitPolicy(radius=1.5)
        self.input_layer = DenseLayer(ob_size, n_hidden, nonlinearity=None)
        if optimizer == 'sgd':
            update_policy = ColumnEiSGD_Clip(max=max_gn)
        elif optimizer == 'adam':
            update_policy = ColumnEiAdam_Clip(max=max_gn)
        self.colcell = ColumnEiCell(n_input=n_hidden, layer_params=(n_hidden, 9),
                            nonlinearity=non_linearity,
                            i2h_init_policy=i2h_init_policy,
                            h2h_init_policy=h2h_init_policy,
                            update_policy=update_policy)
        self.output_layer = DenseLayer(n_hidden, act_size, nonlinearity=None)
        self.entropy_loss = False
    def forward(self, x, hx=None):
        # x: (T, B, D_in)
        T, B, _ = x.shape
        device  = x.device

        # init or unpack hidden/cell: (L, B, H)
        if hx is None:
            h = x.new_zeros(B, self.hidden_size)
            c = x.new_zeros(B, self.hidden_size)
        else:
            h, c = hx

        outputs = torch.zeros(T, B, self.output_size, device=device)  # (T, B, H)
        hs = torch.zeros(T, B, self.hidden_size, device=device)  # (T, B, H)
        cs = torch.zeros(T, B, self.hidden_size, device=device)
        rec_ins = torch.zeros(T, B, self.hidden_size, device=device)
        # loop over time
        self.colcell.reset_hidden(B)
        for t in range(T):
            layer_in = x[t]
            input_gate = self.input_layer(layer_in)
            new_h = self.colcell(input_gate)
            hs[t] = new_h
            outputs[t] = self.output_layer(new_h)

            h = new_h
        return outputs, None, (hs, cs, torch.zeros(T,))
    def update(self, lr, max_grad_norm=10):
        self.input_layer.update(lr=lr, max_grad_norm=max_grad_norm)
        self.colcell.update(lr=lr, max_grad_norm=max_grad_norm)
        self.output_layer.update(lr=lr, max_grad_norm=max_grad_norm)
    def get_actual_weight(self):
        return self.colcell.W
    def compute_lyapunov_spectrum(self, x, hx=None):
        """
        李雅普诺夫谱（Oseledec 批量 QR）
        Args:
            x: (T,B,D_in)
        Returns:
            lambda_i: (H,)   # 按 batch 平均后的谱
            lambda_j: (B,H)  # 每个样本的有限时间指数
        动力学（列向量）:
            h_t = φ(rec_in_t + W h_{t-1} + b),  rec_in_t = input_layer(x_t)
            J_t = diag(φ'(z_t)) @ W
        """
        T, B, _ = x.shape
        device = x.device
        H = self.hidden_size

        # 初始化正交基与累计log伸缩
        Q = torch.eye(H, device=device).unsqueeze(0).repeat(B, 1, 1)  # (B,H,H)
        gamma = torch.zeros(B, H, device=device)                      # (B,H)

        # 确定性推进 ColumnEiCell（不经任何可能的随机/约束分支）
        self.colcell.reset_hidden(B)   # self.colcell.h: (H,B)
        W = self.colcell.W             # (H,H)
        b = self.colcell.b             # (H,1)
        act = self.colcell.nonlinearity

        with torch.no_grad():
            for t in range(T):
                # 1) 计算 rec_in_t 与前激活/新隐状态
                rec_in = self.input_layer(x[t])        # (B,H)
                h_prev = self.colcell.h                # (H,B)
                z_t = rec_in.T + W @ h_prev + b        # (H,B)
                h_t = act(z_t) if act is not None else z_t
                self.colcell.h = h_t

                # 2) J_t Q = diag(φ'(z_t)) @ W @ Q
                A = torch.einsum('ij,bjk->bik', W, Q)          # (B,H,H) 先做 WQ
                act_prime = activation_derivative(act, h_t, z_t)          # (B,H)
                JQ = act_prime.unsqueeze(-1) * A                # (B,H,H) 行缩放=左乘对角阵

                # 3) 批量 QR 重正交化并累计对数伸缩
                Q, R = torch.linalg.qr(JQ)                      # Q:(B,H,H), R:(B,H,H)
                diag_R = torch.diagonal(R, dim1=1, dim2=2)      # (B,H)
                gamma += torch.log(torch.abs(diag_R) + 1e-12)

        lambda_j = gamma / float(T)   # (B,H)
        lambda_i = lambda_j.mean(0)   # (H,)
        return lambda_i, lambda_j
    def state_dict(self):
        return {
            'input_layer': self.input_layer.state_dict(),
            'colcell': self.colcell.state_dict(),
            'output_layer': self.output_layer.state_dict(),
        }
    def load_state_dict(self, state_dict):
        self.input_layer.load_state_dict(state_dict['input_layer'])
        self.colcell.load_state_dict(state_dict['colcell'])
        self.output_layer.load_state_dict(state_dict['output_layer'])
