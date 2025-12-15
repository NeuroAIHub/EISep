from termios import B1152000
import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F

import neurogym as ngym

from dale_lib.dense_layers import DenseLayer, EiDense, ColumnEiDense
from dale_lib.model import Model
from dale_lib.dann_rnn import EiRNNCell
from dale_lib.song_rnn import ColumnEiCell
from dale_lib.update_policies import EiRNN_Adam_UpdatePolicy, ColumnEiAdam_Clip, EiRNN_cSGD_UpdatePolicy, DalesANN_cSGD_UpdatePolicy, ColumnEiSGD_Clip, ColumnEiDenseSGD
from dale_lib.init_policies import EiDenseWeightInit_WexMean, U_TorchInit, W_TorchInit, ColumnEiCell_W_InitPolicy, ColumnEi_FirstCell_U_InitPolicy, EiRNNCell_U_InitPolicy, EiRNNCell_W_InitPolicy, W_UniformInit

class EISepRNN(nn.Module):
    """Custom multi-layer sequence LSTM with separate excitatory/inhibitory neurons.
    Input x: (seq_len, batch_size, feature_dim)
    Outputs:
      - output_seq: (seq_len, batch_size, hidden_size)
      - (h_n, c_n): each (num_layers, batch_size, hidden_size).
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # per-layer input → hidden
        self.rec_filter   = nn.ReLU()
        std = 1/ np.sqrt(hidden_size)
        # if isinstance(self.rec_filter, nn.ReLU):
        #     print('Using ReLU as recurrent filter')
        #     std /= np.sqrt(1/2-1/2/np.pi)
        # elif self.rec_filter is torch.abs:
        #     print('Using absolute as recurrent filter')
        #     std /= np.sqrt(1-2/np.pi)
        # std = 0.1

        # self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_layer = nn.Linear(input_size, hidden_size)


        self.Wrec_free = nn.Parameter(nn.ReLU()(torch.randn(hidden_size, hidden_size)*std))
        # print('Wrec_free1:', self.Wrec_free.device)
        # self.Wrec_free = nn.ReLU()(self.Wrec_free)  # Ensure non-negative recurrent weights
        # print('Wrec_free after ReLU1:', self.Wrec_free.device)
        ### test exponential init
        # self.Wrec_free = nn.Parameter(torch.FloatTensor(np.random.exponential(scale=1/np.sqrt(hidden_size), size=(hidden_size, hidden_size))))
        self.bias = nn.Parameter(torch.randn(hidden_size)*std)
        self.eimask_logit = nn.Parameter(torch.randn(hidden_size, 2)*std)
        # self.eicode_origin = nn.Parameter(torch.randn(hidden_size)*std)
        # self.init_W()
        ### test fixed sign
        # self.eimask_logit = nn.Parameter(torch.zeros(hidden_size, 2))
        # self.eimask_logit.requires_grad = False # test for fixed sign
        # exc_idx = torch.randperm(hidden_size)  # 80% excitatory
        # self.eimask_logit[exc_idx[:int(hidden_size*0.9)], 0] = 1.0  # excitatory
        # self.eimask_logit[exc_idx[int(hidden_size*0.9):], 1] = 1.0  # inhibitory
        # set higher inhibitory weight
        # self.Wrec_free.data[:,exc_idx[int(hidden_size*0.9):]] *= 9


        # map top-layer hidden → output
        self.output_layer = nn.Linear(hidden_size, output_size)
        # self.output_layer = EI_Linear(hidden_size, output_size)
        self.act          = nn.Softplus()


        self.training = True
        ###
        # nn.init.orthogonal_(self.Wrec_free)  # Initialize recurrent weights orthogonally
        # self.Wrec_free.data[:,exc_idx[:int(hidden_size*0.8)]] /= 4.0
        # nn.init.normal_(self.eimask_logit)
        for m in [self.input_layer, self.output_layer]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.normal_(m.bias, mean=0, std=std)
    def init_W(self):

        # self.denom = ((2 * np.pi - 1) / (2 * np.pi)) * self.hidden_size
        # sigma_we = np.sqrt(1 / self.denom)
        # self.Wrec_free = nn.Parameter(torch.FloatTensor(np.random.exponential(scale=sigma_we, size=(self.hidden_size, self.hidden_size))))
        self.Wrec_free = nn.Parameter(torch.FloatTensor(np.random.exponential(scale=1/np.sqrt(self.hidden_size), size=(self.hidden_size, self.hidden_size))))

    def forward(self, temperature, x, hx=None):
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
        cs = torch.zeros(T, B, self.hidden_size, device=device)  # (T, B, H)
        rec_ins = torch.zeros(T, B, self.hidden_size, device=device)
        # loop over time
        for t in range(T):
            layer_in = x[t]  # (B, D_in) or (B, H)
            # no loop over layers
            input_gate = self.input_layer(layer_in)  # (B, H)
            ###
            if self.training:
                mask = F.gumbel_softmax(self.eimask_logit,
                            tau=0.1, hard=False, dim=-1)
                # mask = F.softmax(self.eimask_logit/temperature, dim=-1)
                ei_code = mask[:, 0] - mask[:, 1]
                # ei_code = self.eicode_origin
            else:
                ei_code = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
                # ei_code = torch.where(self.eicode_origin > 0, 1., -1.)
            # if temperature < 0:
            #     ei_code = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
            # else:
            #     mask = (self.eimask_logit/temperature).softmax(dim=-1)  # (H, 2)
            #     ei_code = mask[:, 0] - mask[:, 1]
            Wrec   = self.rec_filter(self.Wrec_free)        # (H, H)
            actual = Wrec @ torch.diag(ei_code)     # (H, H)
            # 3) rec input
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
            hs[t] = new_h  # (B, H)
            cs[t] = new_c  # (B, H)
            rec_ins[t] = rec_in
            h = new_h  # (B, H)
            c = new_c  # (B, H)
            # project top-layer hidden to output
            # outputs.append(self.output_layer(new_h))  # (B, H)
            outputs[t] = self.output_layer(new_h)

        # (T, B, H)
        # output_seq = torch.stack(outputs, dim=0)
        # h_seq = torch.stack(hs, dim=0)  # (T, B, H)
        # c_seq = torch.stack(cs, dim=0)  # (T, B, H)
        output_seq = outputs
        h_seq = hs
        c_seq = cs


        return output_seq, (h_seq, c_seq, rec_ins.abs().mean((1,2)))
        # return outputs, (new_h, new_c)

    def eval(self):
        self.training = False
    def train(self, mode=True):
        self.training = mode

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
            if self.training:
                mask = F.gumbel_softmax(self.eimask_logit,
                            tau=0.1, hard=False, dim=-1)
                ei_code = mask[:, 0] - mask[:, 1]
                # ei_code = self.eicode_origin
            else:
                ei_code = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
                # ei_code = torch.where(self.eicode_origin > 0, 1., -1.)
            # if temperature < 0:
            #     ei_code = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
            # else:
            #     mask = (self.eimask_logit/temperature).softmax(dim=-1)  # (H, 2)
            #     ei_code = mask[:, 0] - mask[:, 1]
            Wrec   = self.rec_filter(self.Wrec_free)        # (H, H)
            actual = Wrec @ torch.diag(ei_code)     # (H, H)
            # 3) rec input
            rec_in = h @ actual.T + self.bias          # (B, H)
            # 4) new hidden
            new_h    = self.act(input_gate + rec_in)                   # (B, H)


            # 计算雅可比作用在当前正交基上：J = diag(phi'(preact)) @ actual.T
            # 先算 A = actual.T @ Q  -> (B, H, H)
            A = torch.einsum("ij,bjk->bik", actual.T, Q)  # (B, H, H)

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
    def get_ei_entropy_loss(self):
        eimask_prob = F.softmax(self.eimask_logit, dim=-1)
        eimask_entropy = -torch.sum(eimask_prob * torch.log(eimask_prob + 1e-10), dim=-1)
        entropy_loss = torch.mean(eimask_entropy)  # Add entropy regularization
        for m in [self.input_layer, self.output_layer]:
            if isinstance(m, EI_Linear):
                entropy_loss += m.get_ei_entropy_loss()
        # entropy_loss = - (100-self.eicode_origin **2).abs().mean()
        # sign_eicode = torch.where(self.eicode_origin>0, 1., -1.)
        # entropy_loss = ((sign_eicode - self.eicode_origin) ** 2).mean()
        return entropy_loss

def load_model(prefix: str, num_h: int, ob_size: int, act_size: int, device: str = "cpu") -> nn.Module:
    """Load a model from a file."""
    if "simple" in prefix:
        model = SimpleNet(num_h, ob_size, act_size).to(device)
    else:
        model = Net(num_h, ob_size, act_size).to(device)
        model.set_temperature(-1)
    model.load_state_dict(torch.load(f"./model/perceptualdecisionmaking/{prefix}.pth", map_location=device))
    model.eval()

    return model

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

class Net(nn.Module):
    def __init__(self, num_h, ob_size, act_size, entropy_loss=True):
        super(Net, self).__init__()
        self.hidden_size = num_h
        self.input_size = ob_size
        self.output_size = act_size
        self.lstm = EISepRNN(ob_size, num_h, act_size)
        self.temperature = None
        self.entropy_loss = entropy_loss
        # self.linear = nn.Linear(num_h, act_size)

    def forward(self, x):
        # print('x:', x.shape)
        out, hidden = self.lstm(self.temperature, x)
        # print('out:', out.shape, 'hidden:', hidden[0].shape, hidden[1].shape)
        # x = self.linear(out)
        return out, self.lstm.eimask_logit, hidden

    def get_actual_weight(self):
        """Get the actual recurrent weight matrix."""
        weight = self.lstm.Wrec_free
        corrected_weight = self.lstm.rec_filter(weight)  # Apply the rectifier
        # mask = F.gumbel_softmax(self.lstm.eimask_logit, tau=0.1, hard=False, dim=-1)
        ei_code = torch.where(self.lstm.eimask_logit[:, 0] > self.lstm.eimask_logit[:, 1], 1., -1.)
        # ei_code = self.lstm.eicode_origin
        actual_weight = corrected_weight @ torch.diag(ei_code)
        # actual_weight = corrected_weight @ torch.diag(mask[:, 0] - mask[:, 1])  # Apply the mask to the weight
        return actual_weight

    def set_temperature(self, temperature: float):
        """Set the temperature for the Gumbel-Softmax sampling."""
        self.temperature = temperature

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return (h0, c0)

    def compute_lyapunov_spectrum(self, x, hx=None):
        """Compute the Lyapunov spectrum of the model."""
        return self.lstm.compute_lyapunov_spectrum(x, hx)

    def get_weight_loss(self):
        weight_loss = 0.0
        weight_loss += self.lstm.Wrec_free.abs().mean()  # L1 regularization on recurrent weights
        weight_loss += self.lstm.input_layer.weight.abs().mean()  # L1 regularization on input weights
        weight_loss += self.lstm.output_layer.weight.abs().mean()  # L1 regular
        return weight_loss

    def get_ei_entropy_loss(self):
        return self.lstm.get_ei_entropy_loss()



class simpleRNN(nn.Module):
    """A simple RNN with a single hidden layer."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(simpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_layer = nn.Linear(input_size, hidden_size)
        std = np.sqrt(1/hidden_size)
        # std = 0.1
        # control variable
        # Wrec_free = nn.ReLU()(torch.randn(hidden_size, hidden_size) * std)
        # mask = torch.where(torch.rand(hidden_size, hidden_size) > 0.5, 1, -1)  # Random mask for sparsity
        # self.Wrec_free = nn.Parameter(Wrec_free * mask.float())  # Apply mask to recurrent weights
        # self.Wrec_free.requires_grad = True
        self.Wrec_free = nn.Parameter(torch.randn(hidden_size, hidden_size) * std)

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
            rec_in = h @ self.Wrec_free + self.bias          # (B, H)
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
    def get_ei_entropy_loss(self):
        entropy_loss = 0.0
        for m in [self.input_layer, self.output_layer]:
            if isinstance(m, EI_Linear):
                entropy_loss += m.get_ei_entropy_loss()
        return entropy_loss


class SimpleNet(nn.Module):
    """A simple RNN with a single hidden layer."""
    def __init__(self, num_h: int, ob_size: int, act_size: int):
        super(SimpleNet, self).__init__()
        self.hidden_size = num_h
        self.input_size = ob_size
        self.output_size = act_size
        self.entropy_loss = False
        self.lstm = simpleRNN(ob_size, num_h, act_size)

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
        return self.lstm.Wrec_free
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
        return self.lstm.get_ei_entropy_loss()


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
        T, B, _ = x.shape
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
