from __future__ import annotations

from calendar import c
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from dale_lib.dense_layers import DenseLayer, EiDense, ColumnEiDense
from dale_lib.dann_rnn import EiRNNCell
from dale_lib.update_policies import EiRNN_Adam_UpdatePolicy, ColumnEiAdam_Clip, EiRNN_cSGD_UpdatePolicy, \
        DalesANN_cSGD_UpdatePolicy, DalesANN_cAdam_UpdatePolicy, ColumnEiSGD_Clip, ColumnEiDenseSGD, ColumnEiDenseAdam
from dale_lib.init_policies import EiDenseWeightInit_WexMean, U_TorchInit, W_TorchInit, ColumnEiCell_W_InitPolicy, ColumnEi_FirstCell_U_InitPolicy, EiRNNCell_U_InitPolicy, EiRNNCell_W_InitPolicy, W_UniformInit
from net.dale_util import activation_derivative
from dale_lib.song_rnn import ColumnEiCell


class DANNCell(nn.Module):
    def __init__(self, ob_size, num_h, act_size, optimizer='sgd'):
        super(DANNCell, self).__init__()
        self.hidden_size = num_h
        self.input_size = ob_size
        self.output_size = act_size
        n_hidden = self.hidden_size
        max_gn = None
        non_linearity = nn.Tanh()
        i2h_init_policy=EiRNNCell_U_InitPolicy()
        h2h_init_policy=EiRNNCell_W_InitPolicy(random=False)
        # self.input_layer = DenseLayer(ob_size, n_hidden, nonlinearity=None)
        # self.danncell = EiRNNCell(n_input=n_hidden, ne=n_hidden, ni_i2h=max(num_h//(9+1),1), ni_h2h=max(num_h//(9+1),1),
        #                     nonlinearity=non_linearity,
        #                     i2h_init_policy=i2h_init_policy,
        #                     h2h_init_policy=h2h_init_policy)
        self.danncell = EiRNNCell(n_input=ob_size, ne=n_hidden, ni_i2h=max(num_h//(9+1),1), ni_h2h=max(num_h//(9+1),1),
                    nonlinearity=non_linearity,
                    i2h_init_policy=i2h_init_policy,
                    h2h_init_policy=h2h_init_policy)
        self.output_layer = EiDense(n_hidden, act_size, max(act_size//(9+1),1), nonlinearity=None, weight_init_policy=EiDenseWeightInit_WexMean(numerator=2))
        if optimizer == 'sgd':
            self.danncell.update_policy = EiRNN_cSGD_UpdatePolicy(max_grad_norm=max_gn)
            self.output_layer.update_policy = DalesANN_cSGD_UpdatePolicy(max_grad_norm=max_gn)
        elif optimizer == 'adam':
            self.danncell.update_policy = EiRNN_Adam_UpdatePolicy(max_grad_norm=max_gn)
            self.output_layer.update_policy = DalesANN_cAdam_UpdatePolicy(max_grad_norm=max_gn)
        # self.output_layer = DenseLayer(n_hidden, act_size, nonlinearity=None)
        self.entropy_loss = False
        self.danncell.init_weights()
        self.output_layer.init_weights()
    def forward(self, x, hx=None):
        # x: (B, D_in)

        # for t in range(T):
        layer_in = x
        # input_gate = self.input_layer(layer_in)
        new_h = self.danncell(layer_in)
        output = self.output_layer(new_h)

        return output, new_h
    def update(self, lr, max_grad_norm=10):
        # self.input_layer.update(lr=lr, max_grad_norm=max_grad_norm)
        self.danncell.update(lr=lr, max_grad_norm=max_grad_norm)
        self.output_layer.update(lr=lr, max_grad_norm=max_grad_norm)
    def get_actual_weight(self):
        return self.danncell.W

    def reset_hidden(self, batch_size):
        self.danncell.reset_hidden(batch_size)
        return self.danncell.h.T

    # def state_dict(self):
    #     return {
    #         'input_layer': self.input_layer.state_dict(),
    #         'danncell': self.danncell.state_dict(),
    #         'output_layer': self.output_layer.state_dict(),
    #     }
    # def load_state_dict(self, state_dict):
    #     self.input_layer.load_state_dict(state_dict['input_layer'])
    #     self.danncell.load_state_dict(state_dict['danncell'])
    #     self.output_layer.load_state_dict(state_dict['output_layer'])

    def reset_hidden_on_done(self, hidden, done_mask):
        assert done_mask.dim() == 2 and done_mask.shape[1] == 1
        self.danncell.h = self.danncell.h * (1 - done_mask.T)
        return self.danncell.h.T

    def detach_hidden(self, hidden):
        self.danncell.h = self.danncell.h.detach()
        return  self.danncell.h.T


class ColEICell(nn.Module):
    def __init__(self, ob_size, num_h, act_size, optimizer='sgd'):
        super(ColEICell, self).__init__()
        self.hidden_size = num_h
        self.input_size = ob_size
        self.output_size = act_size
        n_hidden = self.hidden_size
        max_gn = 10
        non_linearity = nn.Tanh()
        i2h_init_policy=ColumnEi_FirstCell_U_InitPolicy(dataset='PennTreebank')
        h2h_init_policy=ColumnEiCell_W_InitPolicy(radius=1.5)
        # self.input_layer = DenseLayer(ob_size, n_hidden, nonlinearity=None)
        if optimizer == 'sgd':
            update_policy = ColumnEiSGD_Clip(max=max_gn)
            output_layer_update_policy = ColumnEiDenseSGD(max=max_gn)
        elif optimizer == 'adam':
            update_policy = ColumnEiAdam_Clip(max=max_gn)
            output_layer_update_policy = ColumnEiDenseAdam(max=max_gn)
        self.colcell = ColumnEiCell(n_input=self.input_size, layer_params=(n_hidden, 9),
                            nonlinearity=non_linearity,
                            i2h_init_policy=i2h_init_policy,
                            h2h_init_policy=h2h_init_policy,
                            update_policy=update_policy)
        # self.output_layer = DenseLayer(n_hidden, act_size, nonlinearity=None)
        self.output_layer = ColumnEiDense(n_hidden, (self.output_size, 9), nonlinearity=None, update_policy=output_layer_update_policy)
        self.entropy_loss = False
        self.colcell.init_weights()
        self.output_layer.init_weights()
    def forward(self, x, hx=None):

        layer_in = x
        # input_gate = self.input_layer(layer_in)
        new_h = self.colcell(layer_in)
        output = self.output_layer(new_h)

        return output, new_h

    def reset_hidden(self, batch_size):
        self.colcell.reset_hidden(batch_size)
        return self.colcell.h.T
    def update(self, lr, max_grad_norm=10):
        # self.input_layer.update(lr=lr, max_grad_norm=max_grad_norm)
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
    # def state_dict(self):
    #     return {
    #         'input_layer': self.input_layer.state_dict(),
    #         'colcell': self.colcell.state_dict(),
    #         'output_layer': self.output_layer.state_dict(),
    #     }
    # def load_state_dict(self, state_dict):
    #     self.input_layer.load_state_dict(state_dict['input_layer'])
    #     self.colcell.load_state_dict(state_dict['colcell'])
    #     self.output_layer.load_state_dict(state_dict['output_layer'])

    def reset_hidden_on_done(self, hidden, done_mask):
        assert done_mask.dim() == 2 and done_mask.shape[1] == 1
        self.colcell.h = self.colcell.h * (1 - done_mask.T)
        return self.colcell.h.T

    def detach_hidden(self, hidden):
        self.colcell.h = self.colcell.h.detach()
        return self.colcell.h.T


class DBPNCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super(DBPNCell, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True, nonlinearity='tanh')
        self.signs = [1,-1]
        self.frac_ex = 0.8
        self.frac_in = 1-self.frac_ex
        self.index_ranges = [(0,int(hidden_size*self.frac_ex)),(int(hidden_size*self.frac_ex),hidden_size)]
        W_init = self.initialize_weights(hidden_size, self.index_ranges, self.signs)
        self.rnn.weight_hh_l0.data = torch.from_numpy(W_init).float()
        self.fc = nn.Linear(hidden_size, output_size)
        self.n_classes = output_size
        self.entropy_loss = False


    def forward(self, x, hidden):
        # h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        self.device = x.device
        nSamp, inDim = x.shape
        assert hidden.shape == (nSamp, self.hidden_size)
        # ops = torch.zeros(nSamp,self.hidden_size, requires_grad=False).to(self.device)
        # pred = torch.zeros(nSamp,self.n_classes, requires_grad=False).to(self.device)

        ip = torch.unsqueeze(x, 1)
        h0 = torch.unsqueeze(hidden, 0)
        op, ht = self.rnn(ip,h0) ## pass 1 timestep through RNN
        fr_mask = ht.clone()>0
        htt = fr_mask*ht.clone() ## mask firing rate
        ops = op.reshape(nSamp, op.shape[-1])

        ## Final prediction at every timestep
        pred = self.fc(ops)
        # print('pred:', pred.shape)
        return pred, htt.squeeze(0)

    def update(self, optimizer):
        optimizer.step()
        device = self.rnn.weight_hh_l0.device
        w = self.rnn.weight_hh_l0.detach().cpu().numpy()
        mask = self.generate_mask(w, self.index_ranges, self.signs)
        wm = w*mask
        self.rnn.weight_hh_l0.data = torch.from_numpy(wm).float().to(device)
        return

    def get_actual_weight(self):
        return self.rnn.weight_hh_l0


    def initialize_weights(self, N, index_ranges, signs):
        weights = np.zeros((N, N))
        for ii, (start, end) in enumerate(index_ranges):
            limit = 1 / np.sqrt(N) if signs[ii] == 1 else -1/np.sqrt(N)
            weights[:, start:end] = np.random.uniform(0, limit, size=(N, end-start))
        return weights

    def generate_mask(self, weights, index_ranges, signs):
        mask = np.zeros_like(weights)
        for i, (start, end) in enumerate(index_ranges):
            if signs[i] == 1:
                mask[:, start:end] = np.where(weights[:, start:end] > 0, 1, 0)
            elif signs[i] == -1:
                mask[:, start:end] = np.where(weights[:, start:end] < 0, 1, 0)
        return mask

    def reset_hidden_on_done(self, hidden, done_mask):
        assert done_mask.dim() == 2 and done_mask.shape[1] == 1
        return hidden * (1 - done_mask)

    def detach_hidden(self, hidden):
        return hidden.detach()
