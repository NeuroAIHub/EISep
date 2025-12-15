# create a wrapper environment to save episode returns and episode lengths
envs_wrapper = gym.wrappers.vector.RecordEpisodeStatistics(
    envs, buffer_length=n_envs * n_updates
)
# torch.autograd.set_detect_anomaly(True)
torch.manual_seed(42)
np.random.seed(42)
critic_losses = np.zeros(n_updates)
actor_losses = np.zeros(n_updates)
entropies = np.zeros(n_updates)
agent.train()
prefix = configuration
# use tqdm to get a progress bar for training
for sample_phase in tqdm(range(n_updates)):
    # we don't have to reset the envs, they just continue playing
    # until the episode is over and then reset automatically

    # reset lists that collect experiences of an episode (sample phase)
    ep_value_preds = torch.zeros(n_steps_per_update+1, n_envs, device=device)
    ep_rewards = torch.zeros(n_steps_per_update, n_envs, device=device)
    ep_action_log_probs = torch.zeros(n_steps_per_update, n_envs, device=device)
    ep_entropy = torch.zeros(n_steps_per_update, n_envs, device=device)
    masks = torch.zeros(n_steps_per_update, n_envs, device=device)

    # at the start of training reset all envs to get an initial state
    if sample_phase == 0:
        states, info = envs_wrapper.reset(seed=42)
        hidden = torch.zeros((n_envs, hidden_size), device=device)

    # play n steps in our parallel environments to collect data
    hidden = hidden.detach()
    for step in range(n_steps_per_update):
        # select an action A_{t} using S_{t} as input for the agent
        actions, action_log_probs, state_value_preds, entropy, new_hidden = agent.select_action(
            states, hidden
        )

        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
        states, rewards, terminated, truncated, infos = envs_wrapper.step(
            actions.cpu().numpy()
        )

        ep_value_preds[step] = torch.squeeze(state_value_preds)
        ep_rewards[step] = torch.tensor(rewards, device=device)
        ep_action_log_probs[step] = action_log_probs
        ep_entropy[step] = entropy

        # add a mask (for the return calculation later);
        # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
        done = np.logical_or(terminated, truncated)
        masks[step] = torch.tensor(~done, device=device, dtype=torch.float32)

        # masks[step] = torch.tensor([not term for term in terminated])

        # reset hidden state for terminated environments
        done = torch.BoolTensor(done).unsqueeze(1).to(device)
        hidden = torch.where(done, torch.zeros_like(hidden).to(device), new_hidden)
    with torch.no_grad():
        last_v, _, _ = agent.forward(states, hidden)
    ep_value_preds[-1] = last_v  # shape: [n_envs]


    # calculate the losses for actor and critic
    critic_loss, actor_loss = agent.get_losses(
        ep_rewards,
        ep_action_log_probs,
        ep_value_preds,
        ep_entropy,
        masks,
        gamma,
        lam,
        ent_coef,
        device,
    )
    # update the actor and critic networks
    agent.update_parameters(critic_loss, actor_loss)

    # log the losses and entropy
    # critic_losses.append(critic_loss.detach().cpu().numpy())
    # actor_losses.append(actor_loss.detach().cpu().numpy())
    # entropies.append(entropy.detach().mean().cpu().numpy())
    critic_losses[sample_phase] = critic_loss.detach().cpu().numpy()
    actor_losses[sample_phase] = actor_loss.detach().cpu().numpy()
    entropies[sample_phase] = ep_entropy.detach().mean().cpu().numpy()
