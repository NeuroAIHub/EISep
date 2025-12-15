
def train(configuration, n_updates, actor_lr, critic_lr, simple_type, ei_type):
    # environment hyperparams
    n_seed = 5
    n_envs = 10
    n_steps_per_update = 128
    randomize_domain = False

    # agent hyperparams
    gamma = 0.99
    lam = 0.95  # hyperparameter for GAE
    ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)

    hidden_size = 64

    env_name = "LunarLander-v3"
    # configuration = 'eisep_all'  # options: 'simple', 'eisep_actor', 'eisep_critic', 'eisep_all'
    # Note: the actor has a slower learning rate so that the value targets become
    # more stationary and are theirfore easier to estimate for the critic

    # environment setup
    if randomize_domain:
        envs = gym.vector.AsyncVectorEnv(
            [
                lambda: gym.make(
                    "LunarLander-v3",
                    gravity=np.clip(
                        np.random.normal(loc=-10.0, scale=1.0), a_min=-11.99, a_max=-0.01
                    ),
                    enable_wind=np.random.choice([True, False]),
                    wind_power=np.clip(
                        np.random.normal(loc=15.0, scale=1.0), a_min=0.01, a_max=19.99
                    ),
                    turbulence_power=np.clip(
                        np.random.normal(loc=1.5, scale=0.5), a_min=0.01, a_max=1.99
                    ),
                    max_episode_steps=600,
                )
                for i in range(n_envs)
            ]
        )

    else:
        envs = gym.make_vec(env_name, num_envs=n_envs, max_episode_steps=600)


    obs_shape = envs.single_observation_space.shape[0] # for LunarLander
    # obs_shape = envs.single_observation_space.n # for FrozenLake
    action_shape = envs.single_action_space.n # For LunarLander
    print(f"obs shape: {obs_shape}, action shape: {action_shape}")
    # action_shape = envs.single_action_space.shape[0]  # for HalfCheetah
    # print(f"obs shape: {obs_shape}, action shape: {action_shape}")
    # set the device
    use_cuda = False
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    critic_losses = np.zeros((n_seed, n_updates))
    actor_losses = np.zeros((n_seed, n_updates))
    entropies = np.zeros((n_seed, n_updates))
    returns = np.zeros((n_seed, n_updates))
    net_state_dicts = [None] * n_seed
    for seed in range(n_seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        # init the agent
        if 'single' in configuration :
            agent = A2C_singleRNN(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs, hidden_size, configuration,
                    simple_type=simple_type, ei_type=ei_type)
        else:
            agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs, hidden_size, configuration,
                    simple_type=simple_type, ei_type=ei_type)
        # create a wrapper environment to save episode returns and episode lengths
        envs_wrapper = gym.wrappers.vector.RecordEpisodeStatistics(
            envs, buffer_length=n_envs * n_updates
        )

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
                states, info = envs_wrapper.reset(seed=seed)
                hidden = agent.reset_hidden(batch_size=n_envs)


            # play n steps in our parallel environments to collect data

            hidden = agent.detach_hidden(hidden)
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
                masks[step] = torch.tensor([not term for term in terminated])

                # reset hidden state for terminated environments

                hidden = agent.reset_hidden_on_done(new_hidden, terminated)
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
            critic_losses[seed, sample_phase] = critic_loss.detach().cpu().numpy()
            actor_losses[seed, sample_phase] = actor_loss.detach().cpu().numpy()
            entropies[seed, sample_phase] = entropy.detach().mean().cpu().numpy()
            returns[seed, sample_phase] = ep_rewards.mean().cpu().numpy()
        net_state_dicts[seed] = agent.state_dict()
    info = {
        'critic_loss': critic_losses,
        'actor_loss': actor_losses,
        'entropy': entropies,
        'net_state_dict': net_state_dicts,
        'rewards': returns,
    }
    lrname=f'critic_{critic_lr}_actor_{actor_lr}'
    info_path = f'./result/lunarlander/simple_{simple_type}_ei_{ei_type}/{lrname}/{prefix}.pkl'
    os.makedirs(os.path.dirname(info_path), exist_ok=True)
    with open(info_path, 'wb') as f:
        pkl.dump(info, f)
    return info
