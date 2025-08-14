import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch

import neurogym as ngym
from torch.nn import functional as F
from IPython.display import clear_output

warnings.filterwarnings("ignore")
clear_output()


def train(env, net, epochs=2000, lyapunov_every=100, batch_size=32, seq_len=100, device='cuda:0'):
    seq_len = 100
    batch_size = 32
    dataset = ngym.Dataset(env, batch_size=batch_size, seq_len=seq_len)

    net.train()
    # This weighting deprioritizes class 0 while keeping the other 16 classes equally important,
    # aligning with the reward distribution idea from the RL setting
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    loss_history = []
    loss_log = np.zeros((epochs, 3))
    gradient_log = np.zeros((epochs, 1))
    spectral_radius_log = np.zeros((epochs, 1))
    rec_balance_log = np.zeros((epochs, seq_len))  # Track input balance
    n_lyapunov = epochs // lyapunov_every
    lyapunov_spectrum_log = np.zeros((n_lyapunov, net.hidden_size))  # Track Lyapunov spectrum
    np.set_printoptions(threshold=np.inf)

    for i in range(epochs):
        # Get a batch of data
        inputs, targets = dataset()
        # print(inputs[..., -20:].argmax(axis=-1))
        # Convert to PyTorch tensors
        inputs = torch.from_numpy(inputs).float().to(device)
        targets = torch.from_numpy(targets).long().to(device)

        # Initialize hidden state
        # hidden = sl_model.init_hidden(inputs.size(1), device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with hidden state tracking
        outputs, eimask_logit, hidden = net(inputs)

        # clip for decision period
        #TODO: not so necessary in other envs
        outputs = outputs[env.decision_idx:]
        targets = targets[env.decision_idx:]

        # Reshape for CrossEntropyLoss
        outputs_flat = outputs.reshape(-1, outputs.size(2))
        targets_flat = targets.reshape(-1)

        # Calculate loss
        # Weight the loss to account for class imbalance (very low weight to 0s, higher weights to 1s and 2s)
        decoding_loss = criterion(outputs_flat, targets_flat)


        if net.entropy_loss:
            eimask_prob = F.softmax(eimask_logit, dim=-1)
            eimask_entropy = -torch.sum(eimask_prob * torch.log(eimask_prob + 1e-10), dim=-1)
            entropy_loss = torch.mean(eimask_entropy)  # Add entropy regularization
            loss_log[i, 2] = entropy_loss.item()
        else:
            entropy_loss = 0.0



        loss = entropy_loss * 1 + decoding_loss * 1

        # Backward pass and optimize
        loss.backward()
        gradient_log[i, 0] = net.lstm.Wrec_free.grad.norm(2).item()
        optimizer.step()

        # Log spectral radius
        weight = net.get_actual_weight()
        spectral_radius = torch.linalg.eigvals(weight).abs().max().item()
        spectral_radius_log[i, 0] = spectral_radius

        rec_balance_log[i] = hidden[-1].detach().cpu().numpy()

        # print statistics
        loss_history.append(loss.item())
        loss_log[i, 0] = loss.item()
        loss_log[i, 1] = decoding_loss.item()
        if i % lyapunov_every == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(i, epochs, loss.item()))
            # Calculate Lyapunov spectrum
            lyapunov_spectrum, _ = net.compute_lyapunov_spectrum(inputs)
            lyapunov_spectrum_log[i // lyapunov_every] = lyapunov_spectrum.detach().cpu().numpy()


    print('Finished Training')
    return net, loss_log, gradient_log, spectral_radius_log, rec_balance_log, lyapunov_spectrum_log





def plot_loss_curve(loss_log_ei, gradient_log_ei, loss_log_simple, gradient_log_simple,
                    spectral_radius_log_ei=None, spectral_radius_log_simple=None,
                    rec_balance_log_ei=None, rec_balance_log_simple=None):
    fig, ax = plt.subplots(1,4, figsize=(20, 5))
    fontsize = 15
    ax = ax.flatten()
    mean_ei = np.mean(loss_log_ei, axis=0)
    std_ei = np.std(loss_log_ei, axis=0)  # population std; use ddof=1 for sample std
    mean_simple = np.mean(loss_log_simple, axis=0)
    std_simple = np.std(loss_log_simple, axis=0)
    ax[0].plot(mean_ei[:, 0], label='EI', color='C0')
    ax[0].fill_between(np.arange(len(mean_ei)), mean_ei[:, 0] - std_ei[:, 0], mean_ei[:, 0] + std_ei[:, 0], alpha=0.2, color='C0')
    ax[0].plot(mean_simple[:, 0], label='simple', color='C1')
    ax[0].fill_between(np.arange(len(mean_simple)), mean_simple[:, 0] - std_simple[:, 0], mean_simple[:, 0] + std_simple[:, 0], alpha=0.2, color='C1')
    ax[0].set_title('Loss', fontsize=fontsize)
    ax[0].set_xlabel('Epoch', fontsize=fontsize)
    ax[0].legend(fontsize=fontsize)

    mean_ei = np.mean(gradient_log_ei, axis=0)
    std_ei = np.std(gradient_log_ei, axis=0)  # population std; use ddof=1 for sample std
    mean_simple = np.mean(gradient_log_simple, axis=0)
    std_simple = np.std(gradient_log_simple, axis=0)
    log_mean_ei = np.log(mean_ei[:, 0] + 1e-10)  # Add small constant to avoid log(0)
    log_mean_simple = np.log(mean_simple[:, 0] + 1e-10)  # Add small constant to avoid log(0)
    log_std_ei = np.log(std_ei[:, 0] + 1e-10)  # Add small constant to avoid log(0)
    log_std_simple = np.log(std_simple[:, 0] + 1e-10)  # Add small constant to avoid log(0)
    ax[1].plot(log_mean_ei, label='EI', color='C0')
    ax[1].fill_between(np.arange(len(mean_ei)), log_mean_ei - log_std_ei, log_mean_ei + log_std_ei, alpha=0.2, color='C0')
    ax[1].plot(log_mean_simple, label='simple', color='C1')
    ax[1].fill_between(np.arange(len(mean_simple)), log_mean_simple - log_std_simple, log_mean_simple + log_std_simple, alpha=0.2, color='C1')
    ax[1].set_title('Log Gradient norm', fontsize=fontsize)
    ax[1].set_xlabel('Epoch', fontsize=fontsize)
    ax[1].legend(fontsize=fontsize)

    if spectral_radius_log_ei is not None and spectral_radius_log_simple is not None:
        mean_ei = np.mean(spectral_radius_log_ei, axis=0)
        std_ei = np.std(spectral_radius_log_ei, axis=0)
        mean_simple = np.mean(spectral_radius_log_simple, axis=0)
        std_simple = np.std(spectral_radius_log_simple, axis=0)
        ax[2].plot(mean_ei[:, 0], label='EI', color='C0')
        ax[2].fill_between(np.arange(len(mean_ei)), mean_ei[:, 0] - std_ei[:, 0], mean_ei[:, 0] + std_ei[:, 0], alpha=0.2, color='C0')
        ax[2].plot(mean_simple[:, 0], label='simple', color='C1')
        ax[2].fill_between(np.arange(len(mean_simple)), mean_simple[:, 0] - std_simple[:, 0], mean_simple[:, 0] + std_simple[:, 0], alpha=0.2, color='C1')
        ax[2].set_title('Spectral radius', fontsize=fontsize)
        ax[2].set_xlabel('Epoch', fontsize=fontsize)
        ax[2].legend(fontsize=fontsize)

    if rec_balance_log_ei is not None and rec_balance_log_simple is not None:
        mean_ei = np.mean(rec_balance_log_ei, axis=(0,-1))
        std_ei = np.std(rec_balance_log_ei, axis=(0,-1))
        mean_simple = np.mean(rec_balance_log_simple, axis=(0,-1))
        std_simple = np.std(rec_balance_log_simple, axis=(0,-1))
        ax[3].plot(mean_ei, label='EI', color='C0')
        ax[3].fill_between(np.arange(len(mean_ei)), mean_ei - std_ei, mean_ei + std_ei, alpha=0.2, color='C0')
        ax[3].plot(mean_simple, label='simple', color='C1')
        ax[3].fill_between(np.arange(len(mean_simple)), mean_simple - std_simple, mean_simple + std_simple, alpha=0.2, color='C1')
        ax[3].set_title('Recurrent balance', fontsize=fontsize)
        ax[3].set_xlabel('Epoch', fontsize=fontsize)
        ax[3].legend(fontsize=fontsize)

    for a in ax:
        a.tick_params(labelsize=fontsize)

    plt.tight_layout()
    # plt.savefig('./output/multitask/yang19_loss_curve.png')
    plt.show()

def plot_lyapunov_spectrum(lyapunov_spectrum_log_ei, lyapunov_spectrum_log_simple):
    n_lyapunov = lyapunov_spectrum_log_ei.shape[1]
    hidden_size = lyapunov_spectrum_log_ei.shape[2]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fontsize = 15
    mean_ei = np.mean(lyapunov_spectrum_log_ei, axis=0)
    std_ei = np.std(lyapunov_spectrum_log_ei, axis=0)  # population std; use ddof=1 for sample std
    mean_simple = np.mean(lyapunov_spectrum_log_simple, axis=0)
    std_simple = np.std(lyapunov_spectrum_log_simple, axis=0)
    argsort_ei = np.flip(np.argsort(mean_ei, axis=1), axis=1)
    argsort_simple = np.flip(np.argsort(mean_simple, axis=1), axis=1)
    sorted_mean_ei = np.take_along_axis(mean_ei, argsort_ei, axis=1)
    sorted_std_ei = np.take_along_axis(std_ei, argsort_ei, axis=1)
    sorted_mean_simple = np.take_along_axis(mean_simple, argsort_simple, axis=1)
    sorted_std_simple = np.take_along_axis(std_simple, argsort_simple, axis=1)
    alpha_list = np.linspace(0.1, 1.0, n_lyapunov)
    for i in range(n_lyapunov):
        ax[0].plot(sorted_mean_ei[i], label=f'Lyapunov {i+1}', alpha=alpha_list[i], color='C0')
        ax[0].plot(sorted_mean_simple[i], label=f'Lyapunov {i+1}', alpha=alpha_list[i], color='C1')
        # ax[0].fill_between(np.arange(hidden_size), sorted_mean_ei[i] - sorted_std_ei[i], sorted_mean_ei[i] + sorted_std_ei[i], alpha=0.2, color='C0')
        ax[1].plot(sorted_mean_simple[i], label=f'Lyapunov {i+1}', alpha=alpha_list[i], color='C1')
        # ax[1].fill_between(np.arange(hidden_size), sorted_mean_simple[i] - sorted_std_simple[i], sorted_mean_simple[i] + sorted_std_simple[i], alpha=0.2, color='C1')
    # ax[0].set_title('Lyapunov spectrum (EI)', fontsize=fontsize)
    ax[1].set_title('Lyapunov spectrum (simple)', fontsize=fontsize)
    ax[0].set_xlabel('Lyapunov index', fontsize=fontsize)
    ax[1].set_xlabel('Lyapunov index', fontsize=fontsize)
    ax[0].set_ylabel('Lyapunov exponent', fontsize=fontsize)
    ax[1].set_ylabel('Lyapunov exponent', fontsize=fontsize)
    # ax[0].legend(fontsize=fontsize)
    # ax[1].legend(fontsize=fontsize)
    for a in ax:
        a.tick_params(labelsize=fontsize)
    plt.tight_layout()
    # plt.savefig('./output/multitask/yang19_lyapunov_spectrum.png')
    plt.show()

def plot_lyapunov_spectrum_separately(lyapunov_spectrum_log_ei, lyapunov_spectrum_log_simple):
    n_lyapunov = lyapunov_spectrum_log_ei.shape[1]
    hidden_size = lyapunov_spectrum_log_ei.shape[2]
    fig, ax = plt.subplots(4, 5, figsize=(16, 10), sharex=True)
    ax = ax.flatten()
    fontsize = 15
    mean_ei = np.mean(lyapunov_spectrum_log_ei, axis=0)
    std_ei = np.std(lyapunov_spectrum_log_ei, axis=0)  # population std; use ddof=1 for sample std
    mean_simple = np.mean(lyapunov_spectrum_log_simple, axis=0)
    std_simple = np.std(lyapunov_spectrum_log_simple, axis=0)
    argsort_ei = np.flip(np.argsort(mean_ei, axis=1), axis=1)
    argsort_simple = np.flip(np.argsort(mean_simple, axis=1), axis=1)
    sorted_mean_ei = np.take_along_axis(mean_ei, argsort_ei, axis=1)
    sorted_std_ei = np.take_along_axis(std_ei, argsort_ei, axis=1)
    sorted_mean_simple = np.take_along_axis(mean_simple, argsort_simple, axis=1)
    sorted_std_simple = np.take_along_axis(std_simple, argsort_simple, axis=1)
    # alpha_list = np.linspace(0.1, 1.0, n_lyapunov)
    for i in range(n_lyapunov):
        ax[i].plot(sorted_mean_ei[i], label='EI', color='C0')
        ax[i].fill_between(np.arange(hidden_size), sorted_mean_ei[i] - sorted_std_ei[i], sorted_mean_ei[i] + sorted_std_ei[i], alpha=0.2, color='C0')
        ax[i].plot(sorted_mean_simple[i], label='simple', color='C1')
        ax[i].fill_between(np.arange(hidden_size), sorted_mean_simple[i] - sorted_std_simple[i], sorted_mean_simple[i] + sorted_std_simple[i], alpha=0.2, color='C1')
        ax[i].set_title(f'Epoch {i+1}*100', fontsize=fontsize)
        if i > 14:
            ax[i].set_xlabel('Lyapunov index', fontsize=fontsize)
        if i % 5 == 0:
            ax[i].set_ylabel('Lyapunov exponent', fontsize=fontsize)
        ax[i].legend(fontsize=fontsize)
        ax[i].tick_params(labelsize=fontsize)
    plt.tight_layout()
    # plt.savefig('./output/multitask/yang19_lyapunov_spectrum_separately.png')
    plt.show()
