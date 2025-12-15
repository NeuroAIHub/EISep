from termios import B1152000
import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F

from dale_lib.dense_layers import DenseLayer, EiDense, ColumnEiDense
from dale_lib.dann_rnn import EiRNNCell
from dale_lib.update_policies import EiRNN_Adam_UpdatePolicy, ColumnEiAdam_Clip, EiRNN_cSGD_UpdatePolicy, DalesANN_cSGD_UpdatePolicy, ColumnEiSGD_Clip, ColumnEiDenseSGD
from dale_lib.init_policies import EiDenseWeightInit_WexMean, U_TorchInit, W_TorchInit, ColumnEiCell_W_InitPolicy, ColumnEi_FirstCell_U_InitPolicy, EiRNNCell_U_InitPolicy, EiRNNCell_W_InitPolicy, W_UniformInit
from docs.examples.net.dale_util import activation_derivative



class DANN(nn.Module):
    def __init__(self, num_h, ob_size, act_size, optimizer='sgd'):
        super(DANN, self).__init__()
        self.hidden_size = num_h
        self.input_size = ob_size
        self.output_size = act_size
        n_hidden = self.hidden_size
        max_gn = 10
        non_linearity = nn.Softplus()
        i2h_init_policy=EiRNNCell_U_InitPolicy()
        h2h_init_policy=EiRNNCell_W_InitPolicy(random=False)
        self.input_layer = DenseLayer(ob_size, n_hidden, nonlinearity=None)
        self.danncell = EiRNNCell(n_input=n_hidden, ne=n_hidden, ni_i2h=max(num_h//(9+1),1), ni_h2h=max(num_h//(9+1),1),
                            nonlinearity=non_linearity,
                            i2h_init_policy=i2h_init_policy,
                            h2h_init_policy=h2h_init_policy)
        if optimizer == 'sgd':
            self.danncell.update_policy = EiRNN_cSGD_UpdatePolicy(max_grad_norm=max_gn)
        elif optimizer == 'adam':
            self.danncell.update_policy = EiRNN_Adam_UpdatePolicy(max_grad_norm=max_gn)
        self.output_layer = DenseLayer(n_hidden, act_size, nonlinearity=None)
        self.entropy_loss = False
    def forward(self, x, hx=None):
        # x: (T, B, D_in)
        T, B, D_in = x.shape
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
        self.danncell.reset_hidden(B)
        for t in range(T):
            layer_in = x[t]
            input_gate = self.input_layer(layer_in)
            new_h = self.danncell(input_gate)
            hs[t] = new_h
            outputs[t] = self.output_layer(new_h)

            h = new_h

        return outputs, None, (hs, cs, torch.zeros(T,))
    def update(self, lr, max_grad_norm=10):
        self.input_layer.update(lr=lr, max_grad_norm=max_grad_norm)
        self.danncell.update(lr=lr, max_grad_norm=max_grad_norm)
        self.output_layer.update(lr=lr, max_grad_norm=max_grad_norm)
    def get_actual_weight(self):
        return self.danncell.W

    def compute_lyapunov_spectrum(self, x, hx=None):
        """
        计算李雅普诺夫谱（Oseledec 批量 QR）
        Return:
            lambda_i: (H,)   # 按隐藏维度平均后的谱
            lambda_j: (B, H) # 每个样本、每个隐藏维度的有限时间指数
        说明：
            动力学为列向量更新：h_t = φ(W h_{t-1} + x_t + b)
            因此雅可比 J_t = diag(φ'(z_t)) @ W ，不是 W^T。
        """
        # x: (T, B, D_in)
        T, B, D_in = x.shape
        device = x.device
        H = self.hidden_size

        # 1) 初始化正交基与累计量
        Q = torch.eye(H, device=device).unsqueeze(0).repeat(B, 1, 1)  # (B, H, H)
        gamma = torch.zeros(B, H, device=device)                      # (B, H)

        # 2) 准备：确定性推进 EiRNNCell（不走 dropconnect 分支）
        #    重置 cell 的隐藏态使其维度为 (H, B)
        self.danncell.reset_hidden(B)  # self.danncell.h: (H, B)
        W = self.danncell.W            # (H, H) 组合后的实际递归权重
        b = self.danncell.b            # (H, 1)
        act = self.danncell.nonlinearity

        with torch.no_grad():
            for t in range(T):
                # 3) 计算本步前激活 z_t 与新隐藏态 h_t
                # input_layer 输出是 (B, H)，cell 内部用列向量 (H, B)
                x_t = self.input_layer(x[t])          # (B, H)
                h_prev = self.danncell.h              # (H, B)
                z_t = x_t.T + W @ h_prev + b          # (H, B)
                h_t = act(z_t) if act is not None else z_t
                self.danncell.h = h_t                 # (H, B)

                # 4) 雅可比对 Q 的作用：J_t Q = diag(φ'(z_t)) @ W @ Q
                # 先 A = W @ Q  -> (B, H, H)（W 共享，Q 按 batch）
                A = torch.einsum('ij,bjk->bik', W, Q)         # (B, H, H)
                # 再逐样本按行缩放（diag(φ'(z_t))）
                act_prime = activation_derivative(act, h_t, z_t)        # (B, H)
                JQ = act_prime.unsqueeze(-1) * A              # (B, H, H)

                # 5) 批量 QR 重正交化与累计 log|R_ii|
                Q, R = torch.linalg.qr(JQ)                    # Q:(B,H,H), R:(B,H,H)
                diag_R = torch.diagonal(R, dim1=1, dim2=2)    # (B, H)
                gamma += torch.log(torch.abs(diag_R) + 1e-12) # 数值稳定

        # 6) 有限时间指数与平均谱
        lambda_j = gamma / float(T)   # (B, H)
        lambda_i = lambda_j.mean(0)   # (H,)
        return lambda_i, lambda_j
    def state_dict(self):
        return {
            'input_layer': self.input_layer.state_dict(),
            'danncell': self.danncell.state_dict(),
            'output_layer': self.output_layer.state_dict(),
        }
    def load_state_dict(self, state_dict):
        self.input_layer.load_state_dict(state_dict['input_layer'])
        self.danncell.load_state_dict(state_dict['danncell'])
        self.output_layer.load_state_dict(state_dict['output_layer'])
