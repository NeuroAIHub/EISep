import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from net.colei import ColEI
from net.dann import DANN
from net.eisep import EISepNet
from net.simple import SimpleNet, Ccstr_SimpleNet
from net.dbp import DBPN
from net.entryei import EntryEINet

def init_model(ob_size, hidden_size, act_size, model_name):
    if model_name == 'colei_sgd':
        net = ColEI(ob_size=ob_size, num_h=hidden_size, act_size=act_size, optimizer='sgd')
    elif model_name == 'colei_adam':
        net = ColEI(ob_size=ob_size, num_h=hidden_size, act_size=act_size, optimizer='adam')
    elif model_name == 'dann_sgd':
        net = DANN(ob_size=ob_size, num_h=hidden_size, act_size=act_size)
    elif model_name == 'dann_adam':
        net = DANN(ob_size=ob_size, num_h=hidden_size, act_size=act_size, optimizer='adam')
    elif model_name == 'dbpn':
        net = DBPN(ob_size=ob_size, num_h=hidden_size, act_size=act_size)
    elif 'entryei' in model_name:
        if 'fixed' in model_name:
            entropy_loss = False
        else:
            entropy_loss = True
        net = EntryEINet(ob_size=ob_size, num_h=hidden_size, act_size=act_size, entropy_loss=entropy_loss)
    elif 'eisep' in model_name:
        if 'fixed' in model_name:
            entropy_loss = False
        else:
            entropy_loss = True
        net = EISepNet(ob_size=ob_size, num_h=hidden_size, act_size=act_size, entropy_loss=entropy_loss)
    elif 'simple' in model_name:
        if 'ccstr' in model_name:
            net = Ccstr_SimpleNet(ob_size=ob_size, num_h=hidden_size, act_size=act_size)
        else:
            net = SimpleNet(ob_size=ob_size, num_h=hidden_size, act_size=act_size)
    return net
