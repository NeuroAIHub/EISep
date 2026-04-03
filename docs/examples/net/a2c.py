
# Author: Till Zemann
# License: MIT License

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from net.rnncell_variant import SimpleRNNCell, EIRNNCell, SimpleGRUCell, Noshare_EIGRUCell, \
    EIGRUCell, SimpleLSTMCell, EILSTMCell, \
    Fixed_EIRNNCell, EntryEIRNNCell, Fixed_EntryEIRNNCell, \
    Fixed_EILSTMCell, EntryEILSTMCell, Fixed_EntryEILSTMCell, Noshare_EILSTMCell
from net.ffn import FFNCell
from net.rnncell_variant_2 import DANNCell, ColEICell, DBPNCell
from torch import distributions as pyd
from torch.distributions import Normal
from functools import partial



class A2C_singleRNN(nn.Module):
    """
    (Synchronous) Advantage Actor-Critic agent class

    Args:
        n_features: The number of features of the input state.
        n_actions: The number of actions the agent can take.
        device: The device to run the computations on (running on a GPU might be quicker for larger Neural Nets,
                for this code CPU is totally fine).
        critic_lr: The learning rate for the critic network (should usually be larger than the actor_lr).
        actor_lr: The learning rate for the actor network.
        n_envs: The number of environments that run in parallel (on multiple CPUs) to collect experiences.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        device: torch.device,
        critic_lr: float,
        actor_lr: float,
        n_envs: int,
        hidden_size: int,
        configuration: str,
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        # define actor and critic networks
        # self.critic = nn.Sequential(*critic_layers).to(self.device)
        # self.actor = nn.Sequential(*actor_layers).to(self.device)
        # self.critic_actor = nn.Sequential(
        #     nn.Linear(n_features, hidden_size),
        #     nn.Tanh(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.Tanh(),
        #     nn.Linear(hidden_size, n_actions+1)
        # )
        if configuration == 'simple_single':
            print("Using simple RNN for actor and critic")
            self.critic_actor = SimpleRNNCell(n_features, hidden_size, 1+n_actions, extra_mlp=True).to(self.device)
        elif configuration == 'ei_single':
            print("Using EI RNN for critic and simple RNN for actor")
            self.critic_actor = EIRNNCell(n_features, hidden_size, 1+n_actions, extra_mlp=True).to(self.device)
        else:
            raise ValueError("Unknown configuration")

        # define optimizers for actor and critic
        self.critic_actor_optim = optim.Adam(self.critic_actor.parameters(), lr=actor_lr)

    def to(self, device: torch.device) -> A2C:
        """Moves the model to the specified device."""
        self.device = device
        self.critic_actor.to(device)
        return self

    def forward(self, x: np.ndarray, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the networks.

        Args:
            x: A batched vector of states.
            hidden: The hidden state tensor with shape (batch_size, hidden_size).

        Returns:
            state_values: A tensor with the state values, with shape [n_envs,].
            action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
        """
        x = torch.Tensor(x).to(self.device)
        value_logits, hidden = self.critic_actor(x, hidden)
        # value_logits = self.critic_actor(x)
        state_values = value_logits[..., 0]
        action_logits_vec = value_logits[..., 1:]
        return state_values, action_logits_vec, hidden

    def select_action(
        self, x: np.ndarray, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of the chosen actions and the log-probs of those actions.

        Args:
            x: A batched vector of states.
            hidden: The hidden state tensor with shape (batch_size, hidden_size).

        Returns:
            actions: A tensor with the actions, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
            state_values: A tensor with the state values, with shape [n_steps_per_update, n_envs].
        """
        state_values, action_logits, new_hidden = self.forward(x, hidden)
        action_pd = torch.distributions.Categorical(
            logits=action_logits
        )  # implicitly uses softmax
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()
        return actions, action_log_probs, state_values, entropy, new_hidden

    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lam: float,
        ent_coef: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
        using Generalized Advantage Estimation (GAE) to compute the advantages (https://arxiv.org/abs/1506.02438).

        Args:
            rewards: A tensor with the rewards for each time step in the episode, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions taken at each time step in the episode, with shape [n_steps_per_update, n_envs].
            value_preds: A tensor with the state value predictions for each time step in the episode, with shape [n_steps_per_update, n_envs].
            masks: A tensor with the masks for each time step in the episode, with shape [n_steps_per_update, n_envs].
            gamma: The discount factor.
            lam: The GAE hyperparameter. (lam=1 corresponds to Monte-Carlo sampling with high variance and no bias,
                                          and lam=0 corresponds to normal TD-Learning that has a low variance but is biased
                                          because the estimates are generated by a Neural Net).
            device: The device to run the computations on (e.g. CPU or GPU).

        Returns:
            critic_loss: The critic loss for the minibatch.
            actor_loss: The actor loss for the minibatch.
        """
        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device=device)

        # compute the advantages using GAE
        # print(value_preds.shape, value_preds)
        gae = 0.0
        for t in reversed(range(T)):
            td_error = (
                rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # calculate the loss of the minibatch for actor and critic
        # critic_loss = advantages.pow(2).mean()
        critic_loss = F.smooth_l1_loss(value_preds[:-1], (advantages + value_preds[:-1]).detach())

        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # give a bonus for higher entropy to encourage exploration
        actor_loss = (
            -(normalized_advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
        )
        return (critic_loss, actor_loss)

    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        """
        Updates the parameters of the actor and critic networks.

        Args:
            critic_loss: The critic loss.
            actor_loss: The actor loss.
        """
        self.critic_actor_optim.zero_grad()
        loss = critic_loss + actor_loss
        if hasattr(self.critic_actor, 'get_ei_entropy_loss'):
            ei_entropy_loss = self.critic_actor.get_ei_entropy_loss()
            loss += ei_entropy_loss  # weight for EI entropy loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 2)

        self.critic_actor_optim.step()

    def reset_hidden(self, batch_size):
        hidden = None
        if isinstance(self.critic_actor, DANNCell) or isinstance(self.critic_actor, ColEICell):
            hidden = self.critic_actor.reset_hidden(batch_size)
        else:
            hidden = torch.zeros((batch_size, self.critic_actor.hidden_size), device=self.device)
        return hidden

    def reset_hidden_on_done(self, hidden, done):
        """Resets the hidden state tensor for environments that are done."""
        done_tensor = torch.BoolTensor(done).unsqueeze(1).to(self.device)
        # done_tensor = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)
        hidden = self.critic_actor.reset_hidden_on_done(hidden, done_tensor)
        return hidden

    def detach_hidden(self, hidden):
        hidden = self.critic_actor.detach_hidden(hidden)
        return hidden





class A2C(nn.Module):
    """
    (Synchronous) Advantage Actor-Critic agent class

    Args:
        n_features: The number of features of the input state.
        n_actions: The number of actions the agent can take.
        device: The device to run the computations on (running on a GPU might be quicker for larger Neural Nets,
                for this code CPU is totally fine).
        critic_lr: The learning rate for the critic network (should usually be larger than the actor_lr).
        actor_lr: The learning rate for the actor network.
        n_envs: The number of environments that run in parallel (on multiple CPUs) to collect experiences.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        device: torch.device,
        critic_lr: float,
        actor_lr: float,
        n_envs: int,
        hidden_size: int,
        configuration: str,
        simple_type: str,
        ei_type: str
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()
        self.device = device
        self.n_envs = n_envs

        # critic_layers = [
        #     nn.Linear(n_features, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1),  # estimate V(s)
        # ]

        # actor_layers = [
        #     nn.Linear(n_features, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(
        #         32, n_actions
        #     ),  # estimate action logits (will be fed into a softmax later)
        # ]

        # define actor and critic networks
        # self.critic = nn.Sequential(*critic_layers).to(self.device)
        # self.actor = nn.Sequential(*actor_layers).to(self.device)
        assert (simple_type in ['rnn', 'gru', 'lstm', 'linear']) \
            and (ei_type in ['eisep', 'dbpn', 'fixed_eisep', 'entryei', 'fixed_entryei', \
                             'dann_sgd', 'colei_sgd', 'dann_adam', 'colei_adam', 'absolute_eisep',\
                                'eisep_init_ColEI', 'eisep_init_DBPN']), 'No valid configuration'
        if simple_type == 'rnn':
            simple_cell_fn = SimpleRNNCell
            if ei_type == 'eisep':
                ei_cell_fn = EIRNNCell
            elif ei_type == 'eisep_init_ColEI':
                ei_cell_fn = partial(EIRNNCell, init_choice='ColEI')
            elif ei_type == 'eisep_init_DBPN':
                ei_cell_fn = partial(EIRNNCell, init_choice='DBPN')
            elif ei_type == 'absolute_eisep':
                ei_cell_fn = partial(EIRNNCell, rec_filter='abs')
            elif ei_type == 'fixed_eisep':
                ei_cell_fn = Fixed_EIRNNCell
            elif ei_type == 'dbpn':
                ei_cell_fn = DBPNCell
            elif ei_type == 'entryei':
                ei_cell_fn = EntryEIRNNCell
            elif ei_type == 'fixed_entryei':
                ei_cell_fn = Fixed_EntryEIRNNCell
            elif ei_type == 'dann_sgd':
                ei_cell_fn = partial(DANNCell, optimizer='sgd')
            elif ei_type == 'colei_sgd':
                ei_cell_fn = partial(ColEICell, optimizer='sgd')
            elif ei_type == 'dann_adam':
                ei_cell_fn = partial(DANNCell, optimizer='adam')
            elif ei_type == 'colei_adam':
                ei_cell_fn = partial(ColEICell, optimizer='adam')

        elif simple_type == 'gru':
            simple_cell_fn = SimpleGRUCell
            if ei_type == 'eisep':
                ei_cell_fn = EIGRUCell
            elif ei_type == 'dbpn':
                ei_cell_fn = DBPNCell
        elif simple_type == 'lstm':
            simple_cell_fn = SimpleLSTMCell
            if ei_type == 'eisep':
                ei_cell_fn = EILSTMCell
            elif ei_type == 'dbpn':
                ei_cell_fn = DBPNCell
        elif simple_type == 'linear':
            simple_cell_fn = partial(FFNCell, ei_linear=False)
            if ei_type == 'eisep':
                ei_cell_fn = partial(FFNCell, ei_linear=True)
        if configuration == 'simple':
            print(f"Using {str(simple_cell_fn)} RNN for actor and critic")
            self.critic = simple_cell_fn(n_features, hidden_size, 1).to(self.device)
            self.actor = simple_cell_fn(n_features, hidden_size, n_actions).to(self.device)
        elif configuration == 'eisep_critic':
            print(f"Using {str(ei_cell_fn)} for critic and {str(simple_cell_fn)} for actor")
            self.critic = ei_cell_fn(n_features, hidden_size, 1).to(self.device)
            self.actor = simple_cell_fn(n_features, hidden_size, n_actions).to(self.device)
        elif configuration == 'eisep_all':
            print(f"Using {str(ei_cell_fn)} for actor and critic")
            self.critic = ei_cell_fn(n_features, hidden_size, 1).to(self.device)
            self.actor = ei_cell_fn(n_features, hidden_size, n_actions).to(self.device)
        elif configuration == 'eisep_actor':
            print(f"Using {str(simple_cell_fn)} for critic and {str(ei_cell_fn)} for actor")
            self.critic = simple_cell_fn(n_features, hidden_size, 1).to(self.device)
            self.actor = ei_cell_fn(n_features, hidden_size, n_actions).to(self.device)
        else:
            raise ValueError("Unknown configuration")

        # define optimizers for actor and critic
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.ei_entropy_loss_coef = 1.0

    def to(self, device: torch.device) -> A2C:
        """Moves the model to the specified device."""
        self.device = device
        self.critic.to(device)
        self.actor.to(device)
        return self

    def forward(self, x: np.ndarray, hidden: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the networks.

        Args:
            x: A batched vector of states.
            hidden: The hidden state tensor with shape (batch_size, hidden_size).

        Returns:
            state_values: A tensor with the state values, with shape [n_envs,].
            action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
        """
        h_critic, h_actor = hidden
        x = torch.Tensor(x).to(self.device)
        state_values, h_critic = self.critic(x, h_critic)  # shape: [n_envs,]
        action_logits_vec, h_actor = self.actor(x, h_actor)  # shape: [n_envs, n_actions]
        return state_values, action_logits_vec, (h_critic, h_actor)

    def select_action(
        self, x: np.ndarray, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns a tuple of the chosen actions and the log-probs of those actions.

        Args:
            x: A batched vector of states.
            hidden: The hidden state tensor with shape (batch_size, hidden_size).

        Returns:
            actions: A tensor with the actions, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
            state_values: A tensor with the state values, with shape [n_steps_per_update, n_envs].
        """
        state_values, action_logits, new_hidden = self.forward(x, hidden)
        # action_logits = torch.where(torch.isnan(action_logits), 0, action_logits)
        action_pd = torch.distributions.Categorical(
            logits=action_logits
        )  # implicitly uses softmax
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()
        return actions, action_log_probs, state_values, entropy, new_hidden

    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lam: float,
        ent_coef: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
        using Generalized Advantage Estimation (GAE) to compute the advantages (https://arxiv.org/abs/1506.02438).

        Args:
            rewards: A tensor with the rewards for each time step in the episode, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions taken at each time step in the episode, with shape [n_steps_per_update, n_envs].
            value_preds: A tensor with the state value predictions for each time step in the episode, with shape [n_steps_per_update, n_envs].
            masks: A tensor with the masks for each time step in the episode, with shape [n_steps_per_update, n_envs].
            gamma: The discount factor.
            lam: The GAE hyperparameter. (lam=1 corresponds to Monte-Carlo sampling with high variance and no bias,
                                          and lam=0 corresponds to normal TD-Learning that has a low variance but is biased
                                          because the estimates are generated by a Neural Net).
            device: The device to run the computations on (e.g. CPU or GPU).

        Returns:
            critic_loss: The critic loss for the minibatch.
            actor_loss: The actor loss for the minibatch.
        """
        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device=device)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T)):
            td_error = (
                rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        critic_loss = F.smooth_l1_loss(value_preds[:-1], (advantages + value_preds[:-1]).detach())

        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration
        actor_loss = (
            -(normalized_advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
        )
        return (critic_loss, actor_loss)

    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        """
        Updates the parameters of the actor and critic networks.

        Args:
            critic_loss: The critic loss.
            actor_loss: The actor loss.
        """
        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()
        loss = critic_loss + actor_loss
        for m in [self.critic, self.actor]:
            if hasattr(m, 'get_ei_entropy_loss'):
                ei_entropy_loss = m.get_ei_entropy_loss() * self.ei_entropy_loss_coef
                loss += ei_entropy_loss  # weight for EI entropy loss
        loss.backward()
        for m, optim in zip([self.critic, self.actor], [self.critic_optim, self.actor_optim]):

            if isinstance(m, DANNCell):
                m.update(lr=optim.defaults['lr'], max_grad_norm=None)
                m.zero_grad()
            elif isinstance(m, ColEICell):
                m.update(lr=optim.defaults['lr'], max_grad_norm=10)
                m.zero_grad()
            elif isinstance(m, DBPNCell):
                m.update(optim)
            else:
                optim.step()

    def train(self, mode=True) -> None:
        self.critic.train(mode)
        self.actor.train(mode)

    def reset_hidden(self, batch_size):
        hidden = [None, None]
        for i, m in enumerate([self.critic, self.actor]):
            if isinstance(m, DANNCell) or isinstance(m, ColEICell):
                h = m.reset_hidden(batch_size)
            else:
                h = torch.zeros((batch_size, m.hidden_size), device=self.device)
            hidden[i] = h
        return tuple(hidden)

    def reset_hidden_on_done(self, hidden, done):
        """Resets the hidden state tensor for environments that are done."""
        h_critic, h_actor = hidden
        # done_tensor = torch.BoolTensor(done).unsqueeze(1).to(self.device)
        done_tensor = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)
        h_critic = self.critic.reset_hidden_on_done(h_critic, done_tensor)
        h_actor = self.actor.reset_hidden_on_done(h_actor, done_tensor)
        return (h_critic, h_actor)

    def detach_hidden(self, hidden):
        h_critic, h_actor = hidden
        h_critic = self.critic.detach_hidden(h_critic)
        h_actor = self.actor.detach_hidden(h_actor)
        return (h_critic, h_actor)

    def set_temperature(self, temperature, module):
        if module in ['critic', 'both']:
            self.critic.set_temperature(temperature)
        if module in ['actor', 'both']:
            self.actor.set_temperature(temperature)



class ContinuousA2C(nn.Module):
    """
    (Synchronous) Advantage Actor-Critic agent class

    Args:
        n_features: The number of features of the input state.
        n_actions: The number of actions the agent can take.
        device: The device to run the computations on (running on a GPU might be quicker for larger Neural Nets,
                for this code CPU is totally fine).
        critic_lr: The learning rate for the critic network (should usually be larger than the actor_lr).
        actor_lr: The learning rate for the actor network.
        n_envs: The number of environments that run in parallel (on multiple CPUs) to collect experiences.
    """

    def __init__(
        self,
        n_features: int,
        action_dim: int,
        device: torch.device,
        critic_lr: float,
        actor_lr: float,
        n_envs: int,
        hidden_size: int,
        configuration: str,
        simple_type: str,
        ei_type: str
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        self.actor_outsize = action_dim*2
        self.log_std_bounds = (-5, 2)  # bounds for log std of action distribution
        assert (simple_type in ['rnn', 'gru', 'lstm']) \
            and (ei_type in ['eisep', 'dbpn', 'fixed_eisep', 'entryei', 'fixed_entryei', \
                             'dann_sgd', 'colei_sgd', 'dann_adam', 'colei_adam', 'noshare_eisep']), 'No valid configuration'
        if simple_type == 'rnn':
            simple_cell_fn = SimpleRNNCell
            if ei_type == 'eisep':
                ei_cell_fn = EIRNNCell
            elif ei_type == 'fixed_eisep':
                ei_cell_fn = Fixed_EIRNNCell
            elif ei_type == 'dbpn':
                ei_cell_fn = DBPNCell
            elif ei_type == 'entryei':
                ei_cell_fn = EntryEIRNNCell
            elif ei_type == 'fixed_entryei':
                ei_cell_fn = Fixed_EntryEIRNNCell
            elif ei_type == 'dann_sgd':
                ei_cell_fn = partial(DANNCell, optimizer='sgd')
            elif ei_type == 'colei_sgd':
                ei_cell_fn = partial(ColEICell, optimizer='sgd')
            elif ei_type == 'dann_adam':
                ei_cell_fn = partial(DANNCell, optimizer='adam')
            elif ei_type == 'colei_adam':
                ei_cell_fn = partial(ColEICell, optimizer='adam')

        elif simple_type == 'gru':
            simple_cell_fn = SimpleGRUCell
            if ei_type == 'eisep':
                ei_cell_fn = EIGRUCell
            elif ei_type == 'dbpn':
                ei_cell_fn = DBPNCell
        elif simple_type == 'lstm':
            simple_cell_fn = SimpleLSTMCell
            if ei_type == 'eisep':
                ei_cell_fn = EILSTMCell
            elif ei_type == 'noshare_eisep':
                ei_cell_fn = Noshare_EILSTMCell
            elif ei_type == 'fixed_eisep':
                ei_cell_fn = Fixed_EILSTMCell
            elif ei_type == 'entryei':
                ei_cell_fn = EntryEILSTMCell
            elif ei_type == 'fixed_entryei':
                ei_cell_fn = Fixed_EntryEILSTMCell

        if configuration == 'simple':
            print(f"Using {str(simple_cell_fn)} RNN for actor and critic")
            self.critic = simple_cell_fn(n_features, hidden_size, 1).to(self.device)
            self.actor = simple_cell_fn(n_features, hidden_size, self.actor_outsize).to(self.device)
        elif configuration == 'eisep_critic':
            print(f"Using {str(ei_cell_fn)} for critic and {str(simple_cell_fn)} for actor")
            self.critic = ei_cell_fn(n_features, hidden_size, 1).to(self.device)
            self.actor = simple_cell_fn(n_features, hidden_size, self.actor_outsize).to(self.device)
        elif configuration == 'eisep_all':
            print(f"Using {str(ei_cell_fn)} for actor and critic")
            self.critic = ei_cell_fn(n_features, hidden_size, 1).to(self.device)
            self.actor = ei_cell_fn(n_features, hidden_size, self.actor_outsize).to(self.device)
        elif configuration == 'eisep_actor':
            print(f"Using {str(simple_cell_fn)} for critic and {str(ei_cell_fn)} for actor")
            self.critic = simple_cell_fn(n_features, hidden_size, 1).to(self.device)
            self.actor = ei_cell_fn(n_features, hidden_size, self.actor_outsize).to(self.device)
        else:
            raise ValueError("Unknown configuration")

        # define optimizers for actor and critic
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)

    def to(self, device: torch.device) -> A2C:
        """Moves the model to the specified device."""
        self.device = device
        self.critic.to(device)
        self.actor.to(device)
        return self

    # def forward(self, x: np.ndarray, hidden: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    def forward(self, x: np.ndarray, hidden):
        """
        Forward pass of the networks.

        Args:
            x: A batched vector of states.
            hidden: The hidden state tensor with shape (batch_size, hidden_size).

        Returns:
            state_values: A tensor with the state values, with shape [n_envs,].
            action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
        """
        h_critic, h_actor = hidden
        x = torch.Tensor(x).to(self.device)
        state_values, h_critic = self.critic(x, h_critic)  # shape: [n_envs,]
        action_logits_vec, h_actor = self.actor(x, h_actor)  # shape: [n_envs, n_actions]
        mu, log_std = torch.chunk(action_logits_vec, 2, dim=-1)
        std = torch.log(log_std.exp() + 1)
        return state_values, (mu, std), (h_critic, h_actor)

    def init_hidden(self, batch_size: int):
        """Initializes the hidden state tensor."""
        h_critic = self.critic.init_hidden(batch_size)
        h_actor = self.actor.init_hidden(batch_size)
        return (h_critic, h_actor)

    def detach_hidden(self, hidden):
        """Detaches the hidden state tensor from the computation graph."""
        h_critic, h_actor = hidden
        return (self.critic.detach_hidden(h_critic), self.actor.detach_hidden(h_actor))

    def reset_hidden_on_done(self, hidden, done):
        """Resets the hidden state tensor for environments that are done."""
        h_critic, h_actor = hidden
        # done_tensor = torch.BoolTensor(done).unsqueeze(1).to(self.device)
        done_tensor = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)
        h_critic = self.critic.reset_hidden_on_done(h_critic, done_tensor)
        h_actor = self.actor.reset_hidden_on_done(h_actor, done_tensor)
        return (h_critic, h_actor)

    def select_action(
        self, x: np.ndarray, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns a tuple of the chosen actions and the log-probs of those actions.

        Args:
            x: A batched vector of states.
            hidden: The hidden state tensor with shape (batch_size, hidden_size).

        Returns:
            actions: A tensor with the actions, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
            state_values: A tensor with the state values, with shape [n_steps_per_update, n_envs].
        """
        state_values, mu_std, new_hidden = self.forward(x, hidden)
        mu, std = mu_std
        action_pd = Normal(mu, std+1e-5)
        # action_pd = torch.distributions.Categorical(
        #     logits=action_logits
        # )  # implicitly uses softmax
        actions = action_pd.sample()
        # a_env = F.tanh(actions) # squash the actions to be in [-1, 1], TODO: need calibration
        # actions = a_env
        action_log_probs = action_pd.log_prob(actions).sum(-1)
        assert len(actions.shape) == 2, f"actions should be 2D but is {actions.shape}"
        entropy = action_pd.entropy().sum(-1)
        return actions, action_log_probs, state_values, entropy, new_hidden

    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lam: float,
        ent_coef: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
        using Generalized Advantage Estimation (GAE) to compute the advantages (https://arxiv.org/abs/1506.02438).

        Args:
            rewards: A tensor with the rewards for each time step in the episode, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions taken at each time step in the episode, with shape [n_steps_per_update, n_envs].
            value_preds: A tensor with the state value predictions for each time step in the episode, with shape [n_steps_per_update, n_envs].
            masks: A tensor with the masks for each time step in the episode, with shape [n_steps_per_update, n_envs].
            gamma: The discount factor.
            lam: The GAE hyperparameter. (lam=1 corresponds to Monte-Carlo sampling with high variance and no bias,
                                          and lam=0 corresponds to normal TD-Learning that has a low variance but is biased
                                          because the estimates are generated by a Neural Net).
            device: The device to run the computations on (e.g. CPU or GPU).

        Returns:
            critic_loss: The critic loss for the minibatch.
            actor_loss: The actor loss for the minibatch.
        """
        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device=device)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T)):
            td_error = (
                rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        # critic_loss = advantages.pow(2).mean()
        critic_loss = F.smooth_l1_loss(value_preds[:-1], (advantages + value_preds[:-1]).detach())

        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # give a bonus for higher entropy to encourage exploration
        actor_loss = (
            -(normalized_advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
        )
        return (critic_loss, actor_loss)

    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        """
        Updates the parameters of the actor and critic networks.

        Args:
            critic_loss: The critic loss.
            actor_loss: The actor loss.
        """
        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()
        loss = critic_loss + actor_loss
        for m in [self.critic, self.actor]:
            if hasattr(m, 'get_ei_entropy_loss'):
                ei_entropy_loss = m.get_ei_entropy_loss()
                loss += ei_entropy_loss  # weight for EI entropy loss
        loss.backward()
        self.critic_optim.step()
        self.actor_optim.step()

    def train(self, mode=True) -> None:
        self.critic.train(mode)
        self.actor.train(mode)
