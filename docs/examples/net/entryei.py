
from termios import B1152000
import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from net.dale_util import activation_derivative

class Fixed_EntryEIRNN(nn.Module):
    """A simple RNN with a single hidden layer."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        print('Using Ccstr_simpleRNN')
        super(Fixed_EntryEIRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_layer = nn.Linear(input_size, hidden_size)
        std = np.sqrt(1/hidden_size)
        # std = 0.1
        ### control variable Cstr-Simple
        self.mask = nn.Parameter(torch.where(torch.rand(hidden_size, hidden_size) > 0.5, 1., -1.))  # Random mask for sparsity
        self.mask.requires_grad = False  # Mask is not trainable
        self.Wrec_free = nn.Parameter(nn.ReLU()(torch.randn(hidden_size, hidden_size) * std))  # Apply mask to recurrent weights
        self.Wrec_free.requires_grad = True
        ### normal init
        # self.Wrec_free = nn.Parameter(torch.randn(hidden_size, hidden_size) * std)

        # print('Wrec_free:', self.Wrec_free.device)
        # self.Wrec_free = nn.ReLU()(self.Wrec_free)  # Ensure non-negative recurrent weights
        # print('Wrec_free after ReLU:', self.Wrec_free.device)
        self.bias = nn.Parameter(torch.randn(hidden_size) * std)
        self.act = nn.Softplus()
        self.output_layer = nn.Linear(hidden_size, output_size)
        # self.output_layer = EI_Linear(hidden_size, output_size)
        # nn.init.orthogonal_(self.Wrec_free)  # Initialize recurrent weights orthogonally
        for m in [self.input_layer, self.output_layer]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.normal_(m.bias, mean=0, std=std)

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

        outputs = []
        hs = []
        cs = []
        rec_ins = torch.zeros(T, B, self.hidden_size, device=device)
        # loop over time
        for t in range(T):
            layer_in = x[t]  # (B, D_in) or (B, H)
            # no loop over layers
            input_gate = self.input_layer(layer_in)  # (B, H)

            # 3) rec input
            actual = nn.ReLU()(self.Wrec_free) * self.mask.float()  # Apply ReLU to ensure non-negativity
            rec_in = h @ actual.T + self.bias          # (B, H)
            # 4) new hidden
            new_h    = self.act(input_gate + rec_in)                   # (B, H)

            # new_h.append(h_i)
            # new_c.append(c[i])   # 如果以后要更新 c_i，在这里替换
            # layer_in = h_i       # feed to next layer
            new_c = c


            # stack per-layer states
            # h = torch.stack(new_h, dim=0)  # (L, B, H)
            # c = torch.stack(new_c, dim=0)
            hs.append(new_h)  # (B, H)
            cs.append(new_c)  # (B, H)
            rec_ins[t] = rec_in
            h = new_h  # (B, H)
            c = new_c  # (B, H)
            # project top-layer hidden to output
            outputs.append(self.output_layer(new_h))  # (B, H)

        # (T, B, H)
        output_seq = torch.stack(outputs, dim=0)
        h_seq = torch.stack(hs, dim=0)  # (T, B, H)
        c_seq = torch.stack(cs, dim=0)  # (T, B, H)
        return output_seq, (h_seq, c_seq, rec_ins.abs().mean((1,2)))



    def compute_lyapunov_spectrum(self, x, hx=None):
        """Return:
        lambda_i: (H,) # average Lyapunov exponent for each hidden unit
        lambda_j: (B, H) # Lyapunov exponent for each hidden unit in each sample.
        """
        # x: (T, B, D_in)
        T, B, _ = x.shape
        device  = x.device

        # init or unpack hidden/cell: (L, B, H)
        if hx is None:
            h = x.new_zeros(B, self.hidden_size)
            c = x.new_zeros(B, self.hidden_size)
        else:
            h, c = hx


        # 初始化正交基 Q: (B, H, H)
        Q = torch.eye(self.hidden_size, device=device).unsqueeze(0).repeat(B, 1, 1)  # 每个 sample 一组基
        # 累计 gamma: (B, H)
        gamma = torch.zeros(B, self.hidden_size, device=device)

        # 逐步推进
        for t in range(T):
            layer_in = x[t]  # (B, D_in) or (B, H)
            # no loop over layers
            input_gate = self.input_layer(layer_in)  # (B, H)

            # 3) rec input
            rec_in = h @ self.Wrec_free + self.bias          # (B, H)
            # 4) new hidden
            new_h    = self.act(input_gate + rec_in)                   # (B, H)



            # 计算雅可比作用在当前正交基上：J = diag(phi'(preact)) @ actual.T
            # 先算 A = actual.T @ Q  -> (B, H, H)
            A = torch.einsum("ij,bjk->bik", self.Wrec_free.T, Q)  # (B, H, H)

            # 再算 act'（每个 batch, 每个维度）
            act_prime = activation_derivative(self.act, new_h, input_gate+rec_in)  # (B, H)
            # JQ = diag(act_prime) @ A; diag multiplies每一行
            JQ = act_prime.unsqueeze(-1) * A  # (B, H, H)

            # QR 重正交化（batch 版）
            # torch.linalg.qr 在新版 PyTorch 支持 batch
            Q_new, R = torch.linalg.qr(JQ)  # Q_new: (B,H,H), R: (B,H,H)
            # 累加 log 伸缩因子
            diag_R = torch.diagonal(R, dim1=1, dim2=2)  # (B, H)
            gamma += torch.log(torch.abs(diag_R) + 1e-10)  # 防止 0

            # 更新状态 / 基
            Q = Q_new
            h = new_h
            # c 保持不变（和你的 forward 一样）

        # 该 batch 每个 sample 的有限时间 Lyapunov 指数
        lambda_j = gamma / float(T)  # (B, H)
        # 平均得到谱
        lambda_i = lambda_j.mean(dim=0)  # (H,)

        return lambda_i, lambda_j
    def get_actual_weight(self):
        """Get the actual recurrent weight matrix."""
        actual_Wrec = nn.ReLU()(self.Wrec_free) * self.mask.float()
        return actual_Wrec


class EntryEIRNN(nn.Module):
    """A simple RNN with a single hidden layer."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        print('Using Unfixed EntryEI RNN')
        super(EntryEIRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_layer = nn.Linear(input_size, hidden_size)
        std = np.sqrt(1/hidden_size)
        # std = 0.1
        ### control variable Cstr-Simple
        self.eimask_logit = nn.Parameter(torch.rand(hidden_size, hidden_size, 2)*std)  # Random mask for sparsity
        self.eimask_logit.requires_grad = True  # Mask is trainable
        self.Wrec_free = nn.Parameter(nn.ReLU()(torch.randn(hidden_size, hidden_size) * std))  # Apply mask to recurrent weights
        self.Wrec_free.requires_grad = True
        ### normal init
        # self.Wrec_free = nn.Parameter(torch.randn(hidden_size, hidden_size) * std)

        # print('Wrec_free:', self.Wrec_free.device)
        # self.Wrec_free = nn.ReLU()(self.Wrec_free)  # Ensure non-negative recurrent weights
        # print('Wrec_free after ReLU:', self.Wrec_free.device)
        self.bias = nn.Parameter(torch.randn(hidden_size) * std)
        self.act = nn.Softplus()
        self.output_layer = nn.Linear(hidden_size, output_size)
        # self.output_layer = EI_Linear(hidden_size, output_size)
        # nn.init.orthogonal_(self.Wrec_free)  # Initialize recurrent weights orthogonally
        for m in [self.input_layer, self.output_layer]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.normal_(m.bias, mean=0, std=std)

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

        outputs = []
        hs = []
        cs = []
        rec_ins = torch.zeros(T, B, self.hidden_size, device=device)
        # loop over time
        for t in range(T):
            layer_in = x[t]  # (B, D_in) or (B, H)
            # no loop over layers
            input_gate = self.input_layer(layer_in)  # (B, H)
            if self.training:
                mask = F.gumbel_softmax(self.eimask_logit,
                            tau=0.1, hard=False, dim=-1)
                # mask = F.softmax(self.eimask_logit/temperature, dim=-1)
                ei_code = mask[..., 0] - mask[..., 1]
                # ei_code = self.eicode_origin
            else:
                ei_code = torch.where(self.eimask_logit[..., 0] > self.eimask_logit[..., 1], 1., -1.)
            # 3) rec input
            actual = nn.ReLU()(self.Wrec_free) * ei_code  # Apply ReLU to ensure non-negativity
            rec_in = h @ actual.T + self.bias          # (B, H)
            # 4) new hidden
            new_h    = self.act(input_gate + rec_in)                   # (B, H)

            # new_h.append(h_i)
            # new_c.append(c[i])   # 如果以后要更新 c_i，在这里替换
            # layer_in = h_i       # feed to next layer
            new_c = c


            # stack per-layer states
            # h = torch.stack(new_h, dim=0)  # (L, B, H)
            # c = torch.stack(new_c, dim=0)
            hs.append(new_h)  # (B, H)
            cs.append(new_c)  # (B, H)
            rec_ins[t] = rec_in
            h = new_h  # (B, H)
            c = new_c  # (B, H)
            # project top-layer hidden to output
            outputs.append(self.output_layer(new_h))  # (B, H)

        # (T, B, H)
        output_seq = torch.stack(outputs, dim=0)
        h_seq = torch.stack(hs, dim=0)  # (T, B, H)
        c_seq = torch.stack(cs, dim=0)  # (T, B, H)
        return output_seq, (h_seq, c_seq, rec_ins.abs().mean((1,2)))
    def get_ei_entropy_loss(self):
        """Get the EI entropy loss."""
        mask_prob = F.softmax(self.eimask_logit, dim=-1)
        ei_entropy = - (mask_prob * torch.log(mask_prob + 1e-10)).sum(-1).mean()
        return ei_entropy
    def get_actual_weight(self):
        """Get the actual recurrent weight matrix."""
        ei_code = torch.where(self.eimask_logit[..., 0] > self.eimask_logit[..., 1], 1., -1.)
        actual_Wrec = nn.ReLU()(self.Wrec_free) * ei_code
        return actual_Wrec



class EntryEINet(nn.Module):
    """A simple RNN with a single hidden layer."""
    def __init__(self, num_h: int, ob_size: int, act_size: int, entropy_loss: bool = False):
        super(EntryEINet, self).__init__()
        print('Using EntryEINet')
        self.hidden_size = num_h
        self.input_size = ob_size
        self.output_size = act_size
        self.entropy_loss = entropy_loss
        # self.lstm = simpleRNN(ob_size, num_h, act_size)
        ### use Ccstr_simpleRNN
        if entropy_loss == False:
            print('Without EI entropy loss')
            self.lstm = Fixed_EntryEIRNN(ob_size, num_h, act_size)
        else:
            print('With EI entropy loss')
            self.lstm = EntryEIRNN(ob_size, num_h, act_size)

    def forward(self, x):
        # x: (T, B, D_in)
        out, hidden = self.lstm(x)
        # print('history shape:', history.shape, 'hidden shape:', hidden.shape)
        return out, None, hidden
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return (h0, c0)
    def get_actual_weight(self):
        """Get the actual recurrent weight matrix."""
        return self.lstm.get_actual_weight()
    def compute_lyapunov_spectrum(self, x, hx=None):
        """Compute the Lyapunov spectrum of the model."""
        return self.lstm.compute_lyapunov_spectrum(x, hx)
    def get_weight_loss(self):
        weight_loss = 0.0
        weight_loss += self.lstm.Wrec_free.abs().mean()
        weight_loss += self.lstm.input_layer.weight.abs().mean()
        weight_loss += self.lstm.output_layer.weight.abs().mean()
        return weight_loss
    def get_ei_entropy_loss(self):
        if self.entropy_loss == False:
            return 0.0
        return self.lstm.get_ei_entropy_loss()
