import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F

class DBPN(nn.Module):
    def __init__(self, ob_size, num_h, act_size):

        super(DBPN, self).__init__()
        self.hidden_size = num_h
        self.rnn = nn.RNN(ob_size, num_h, num_layers=1, batch_first=True, nonlinearity='tanh')
        self.signs = [1,-1]
        self.frac_ex = 0.8
        self.frac_in = 1-self.frac_ex
        self.index_ranges = [(0,int(num_h*self.frac_ex)),(int(num_h*self.frac_ex),num_h)]
        W_init = initialize_weights(num_h, self.index_ranges, self.signs)
        self.rnn.weight_hh_l0.data = torch.from_numpy(W_init).float().to(self.rnn.weight_hh_l0.device)
        self.fc = nn.Linear(num_h, act_size)
        self.n_classes = act_size
        self.entropy_loss = False


    def forward(self, x):
        x = torch.transpose(x, 0, 1) # [T, B, D_in] -> [B, T, D_in]
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        self.device = x.device
        nSamp, nSteps, inDim = x.shape

        ops = torch.zeros(nSamp,nSteps,self.hidden_size, requires_grad=False).to(self.device)
        pred = torch.zeros(nSamp,nSteps,self.n_classes, requires_grad=False).to(self.device)

        for ii in range(nSteps):
            ip = torch.unsqueeze(x[:,ii,:],1)
            op, ht = self.rnn(ip,h0) ## pass 1 timestep through RNN
            fr_mask = ht.clone()>0
            htt = fr_mask*ht.clone() ## mask firing rate
            h0 = htt ## update hidden state
            ops[:,ii,:] = torch.squeeze(op.clone())

            ## Final prediction at every timestep
            pred[:,ii,:] = self.fc(ops[:, ii, :].clone())
        out = torch.transpose(pred, 0, 1) # [B, T, D_out] -> [T, B, D_out]
        return out, None, (htt, htt, torch.zeros((1,)))

    def update(self, optimizer):
        optimizer.step()
        device = self.rnn.weight_hh_l0.device
        w = self.rnn.weight_hh_l0.detach().cpu().numpy()
        mask = generate_mask(w, self.index_ranges, self.signs)
        wm = w*mask
        self.rnn.weight_hh_l0.data = torch.from_numpy(wm).float().to(device)
        return

    def get_actual_weight(self):
        return self.rnn.weight_hh_l0


def initialize_weights(N, index_ranges, signs):
    weights = np.zeros((N, N))
    for ii, (start, end) in enumerate(index_ranges):
        limit = 1 / np.sqrt(N) if signs[ii] == 1 else -1/np.sqrt(N)
        weights[:, start:end] = np.random.uniform(0, limit, size=(N, end-start))
    return weights

def generate_mask(weights, index_ranges, signs):
    mask = np.zeros_like(weights)
    for i, (start, end) in enumerate(index_ranges):
        if signs[i] == 1:
            mask[:, start:end] = np.where(weights[:, start:end] > 0, 1, 0)
        elif signs[i] == -1:
            mask[:, start:end] = np.where(weights[:, start:end] < 0, 1, 0)
    return mask
