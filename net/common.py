import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from net.colei import ColEI
from net.dann import DANN
from net.eisep import EISepNet
from net.simple import SimpleNet, Ccstr_SimpleNet, Cstr_SimpleNet
from net.dbp import DBPN
from net.entryei import EntryEINet
import os

def init_model(ob_size, hidden_size, act_size, model_name, ei_ratio=None):
    if model_name == 'colei_sgd':
        net = ColEI(ob_size=ob_size, num_h=hidden_size, act_size=act_size, optimizer='sgd')
    elif model_name == 'colei_adam':
        net = ColEI(ob_size=ob_size, num_h=hidden_size, act_size=act_size, optimizer='adam')
    elif model_name == 'dann_sgd':
        net = DANN(ob_size=ob_size, num_h=hidden_size, act_size=act_size, optimizer='sgd')
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
        if model_name == 'absolute_eisep':
            net = EISepNet(ob_size=ob_size, num_h=hidden_size, act_size=act_size, entropy_loss=entropy_loss, rec_filter='abs')
        elif '_ei_' in model_name: ### Added to handle eisep with ei_linear_block options
            if model_name == 'eisep_ei_input':
                ei_linear_block = ['input']
            elif model_name == 'eisep_ei_output':
                ei_linear_block = ['output']
            elif model_name == 'eisep_ei_both':
                ei_linear_block = ['input', 'output']
            else:
                raise ValueError(f"Unknown model_name {model_name}")
            net = EISepNet(ob_size=ob_size, num_h=hidden_size, act_size=act_size, entropy_loss=entropy_loss, ei_linear_block=ei_linear_block)
        elif '_init_' in model_name: ### Added to handle eisep with different initialization schemes
            init_choice = model_name.split('_')[-1]  # get the initialization choice
            net = EISepNet(ob_size=ob_size, num_h=hidden_size, act_size=act_size, entropy_loss=entropy_loss, init_choice=init_choice, ei_ratio=ei_ratio)
        else:
            net = EISepNet(ob_size=ob_size, num_h=hidden_size, act_size=act_size, entropy_loss=entropy_loss)

    elif 'simple' in model_name:
        if 'ccstr' in model_name:
            net = Ccstr_SimpleNet(ob_size=ob_size, num_h=hidden_size, act_size=act_size)
        elif 'cstr' in model_name:
            net = Cstr_SimpleNet(ob_size=ob_size, num_h=hidden_size, act_size=act_size)
        elif '_ei_' in model_name:
            if model_name == 'simple_ei_input':
                ei_linear_block = ['input']
            elif model_name == 'simple_ei_output':
                ei_linear_block = ['output']
            elif model_name == 'simple_ei_both':
                ei_linear_block = ['input', 'output']
            else:
                raise ValueError(f"Unknown model_name {model_name}")
            net = SimpleNet(ob_size=ob_size, num_h=hidden_size, act_size=act_size, ei_linear_block=ei_linear_block)
        else:
            net = SimpleNet(ob_size=ob_size, num_h=hidden_size, act_size=act_size)
    else:
        raise ValueError(f"Unknown model_name {model_name}")
    return net

def savefig(figpath, dpi=400):
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    plt.savefig(figpath, dpi=dpi)
    print(figpath)
