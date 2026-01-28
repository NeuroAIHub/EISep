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
from net.dale_util import activation_derivative


class EISepRNN(nn.Module):
    """Custom multi-layer sequence LSTM with separate excitatory/inhibitory neurons.
    Input x: (seq_len, batch_size, feature_dim)
    Outputs:
      - output_seq: (seq_len, batch_size, hidden_size)
      - (h_n, c_n): each (num_layers, batch_size, hidden_size).
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, rec_filter: str='relu',
                 ei_linear_block = None, init_choice = None):
        print('Using UnFixed EISep RNN')
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # per-layer input → hidden
        # self.rec_filter_fn   = nn.ReLU()
        if rec_filter == 'relu':
            self.rec_filter = nn.ReLU()
        elif rec_filter == 'abs':
            self.rec_filter = torch.abs
        # self.rec_filter_fn = torch.abs
        self.std = std = 1/ np.sqrt(hidden_size)
        # if isinstance(self.rec_filter, nn.ReLU):
        #     print('Using ReLU as recurrent filter')
        #     std /= np.sqrt(1/2-1/2/np.pi)
        # elif self.rec_filter is torch.abs:
        #     print('Using absolute as recurrent filter')
        #     std /= np.sqrt(1-2/np.pi)
        # std = 0.1

        # self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_layer = nn.Linear(input_size, hidden_size)


        self.Wrec_free = nn.Parameter(self.rec_filter(torch.randn(hidden_size, hidden_size)*std))
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

        if ei_linear_block is not None:
            if 'output' in ei_linear_block:
                self.output_layer = EI_Linear(hidden_size, output_size)
            if 'input' in ei_linear_block:
                self.input_layer = EI_Linear(input_size, hidden_size)

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
        if init_choice == 'ColEI':
            print('Using ColEI initialization')
            self.init_weights_ColEI()
        elif init_choice == 'DBPN':
            print('Using DBPN initialization')
            self.init_weights_DBPN()

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
        print('Using DBPN initialization')
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

    def sc_forward(self, x, sparsity_mask, hx=None, temperature=None):
        # print(f'sparsity controlled forward, density={sparsity_mask.float().mean()}')
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
            assert sparsity_mask.shape == Wrec.shape
            actual = Wrec @ torch.diag(ei_code) * sparsity_mask     # (H, H)
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



class EISepNet(nn.Module):
    def __init__(self, num_h, ob_size, act_size, entropy_loss=True, rec_filter='relu', ei_linear_block=None,
                 init_choice=None):
        super(EISepNet, self).__init__()
        self.hidden_size = num_h
        self.input_size = ob_size
        self.output_size = act_size
        if entropy_loss == True:
            self.lstm = EISepRNN(ob_size, num_h, act_size, rec_filter, ei_linear_block=ei_linear_block, init_choice=init_choice)
        else:
            self.lstm = FixedEISepRNN(ob_size, num_h, act_size)
        self.temperature = None
        self.entropy_loss = entropy_loss
        # self.linear = nn.Linear(num_h, act_size)

    def forward(self, x):
        # print('x:', x.shape)
        out, hidden = self.lstm(self.temperature, x)
        # print('out:', out.shape, 'hidden:', hidden[0].shape, hidden[1].shape)
        # x = self.linear(out)
        return out, self.lstm.eimask_logit, hidden
    def sc_forward(self, x, sparsity_mask):
        out, hidden = self.lstm.sc_forward(x=x, sparsity_mask=sparsity_mask)
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
        if hasattr(self.lstm, 'get_ei_entropy_loss'):
            return self.lstm.get_ei_entropy_loss()
        else:
            return 0.0



class FixedEISepRNN(nn.Module):
    """Custom multi-layer sequence LSTM with separate excitatory/inhibitory neurons.
    Input x: (seq_len, batch_size, feature_dim)
    Outputs:
      - output_seq: (seq_len, batch_size, hidden_size)
      - (h_n, c_n): each (num_layers, batch_size, hidden_size).
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        print('Using Fixed EISep RNN')
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # per-layer input → hidden
        self.rec_filter   = nn.ReLU()
        std = 1/ np.sqrt(hidden_size)
        self.input_layer = nn.Linear(input_size, hidden_size)

        self.Wrec_free = nn.Parameter(nn.ReLU()(torch.randn(hidden_size, hidden_size)*std))
        self.bias = nn.Parameter(torch.randn(hidden_size)*std)
        self.eimask_logit = nn.Parameter(torch.randn(hidden_size, 2)*std)
        self.eimask_logit.requires_grad = False # fixed sign

        self.output_layer = nn.Linear(hidden_size, output_size)
        # self.output_layer = EI_Linear(hidden_size, output_size)
        self.act          = nn.Softplus()

        self.training = True
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
            ei_code = torch.where(self.eimask_logit[:, 0] > self.eimask_logit[:, 1], 1., -1.)
            Wrec   = self.rec_filter(self.Wrec_free)        # (H, H)
            actual = Wrec @ torch.diag(ei_code)     # (H, H)
            # 3) rec input
            rec_in = h @ actual.T + self.bias          # (B, H)
            # 4) new hidden
            new_h    = self.act(input_gate + rec_in)                   # (B, H)
            new_c = c
            hs[t] = new_h  # (B, H)
            cs[t] = new_c  # (B, H)
            rec_ins[t] = rec_in
            h = new_h  # (B, H)
            c = new_c  # (B, H)
            outputs[t] = self.output_layer(new_h)
        output_seq = outputs
        h_seq = hs
        c_seq = cs

        return output_seq, (h_seq, c_seq, rec_ins.abs().mean((1,2)))
