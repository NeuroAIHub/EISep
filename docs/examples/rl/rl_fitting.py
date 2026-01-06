import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from net.dale_util import RL_Hidden_Buffer
from rl.rl_util import tag_lunar_lander_phases_batch_2, mutual_information
from sklearn.metrics import adjusted_mutual_info_score


class GMM():
    def __init__(self):
        pass

    def fit_one_K(self, Z, K, seed=0):
        """
        Z: [T, r] mode coefficients (recommended) or PCA features.
        Returns: labels [T], best_gmm
        """
        gmm = GaussianMixture(n_components=K, covariance_type="full", random_state=seed)
        gmm.fit(Z)
        return gmm

    def cluster_hidden_regimes(self, Z, K_min=2, K_max=12, seed=0):
        """
        Z: [T, r] mode coefficients (recommended) or PCA features.
        Returns: labels [T], best_gmm
        """
        Zs = StandardScaler().fit_transform(Z)

        best_model = None
        best_bic = np.inf
        for K in range(K_min, K_max+1):
            gmm = self.fit_one_K(Zs, K, seed=seed)
            bic = gmm.bic(Zs)
            if bic < best_bic:
                best_bic = bic
                best_model = gmm
        print('best K:', best_model.n_components, 'BIC:', best_bic)
        labels = best_model.predict(Zs)
        return labels, best_model

    def interpret_clusters(self, labels, buffer_state, buffer_action, n_actions=4):
        K = int(labels.max()) + 1
        action = buffer_action.reshape(-1).astype(int)

        summaries = []
        for k in range(K):
            idx = (labels == k)
            nk = idx.sum()
            if nk < 10:
                continue

            s = buffer_state[idx]
            a = action[idx]

            state_mean = s.mean(axis=0)
            state_std  = s.std(axis=0)

            # action distribution
            act_counts = np.bincount(a, minlength=n_actions)
            act_prob = act_counts / (act_counts.sum() + 1e-12)

            summaries.append({
                "cluster": k,
                "count": int(nk),
                "state_mean": state_mean,
                "state_std": state_std,
                "action_prob": act_prob,
            })
        return summaries

    def transition_matrix(self, labels):
        K = int(labels.max()) + 1
        T = np.zeros((K, K), dtype=np.float64)
        for t in range(len(labels)-1):
            T[labels[t], labels[t+1]] += 1
        T = T / (T.sum(axis=1, keepdims=True) + 1e-12)
        return T




def fit_hmm_gaussian(Z, done, n_components=8, cov_type="full", n_iter=200, seed=0):
    """
    Z: [T, r] latent features (mode coefficients / PCA of hidden)
    done: [T] bool/int, 1 at terminal steps (episode boundary)
    """
    Z = np.asarray(Z)
    done = np.asarray(done).reshape(-1).astype(bool)

    # Build lengths: split at done==1
    ends = np.where(done)[0]
    # Assume done marks the last step of an episode
    lengths = []
    start = 0
    for e in ends:
        L = (e - start + 1)
        if L > 1:
            lengths.append(L)
        start = e + 1
    if start < len(done):  # trailing partial episode
        lengths.append(len(done) - start)

    # Scale features
    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z)

    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type=cov_type,   # "full" or "diag"
        n_iter=n_iter,
        random_state=seed,
        verbose=False
    )

    model.fit(Zs, lengths=lengths)

    # Viterbi labels for all timesteps
    labels = model.predict(Zs, lengths=lengths)  # [T]
    post = model.predict_proba(Zs, lengths=lengths)  # [T, K], optional

    return labels, post, model, scaler, lengths


def run_hmm_metrics(simple_type='rnn', ei_type='eisep', configuration='eisep_all', seed=None,
                    actor_lr=0.001, critic_lr=0.005, network_name='critic', n_components=5,
                    record=False):
    buffer = RL_Hidden_Buffer(10000)

    info_path = f'./rl/result/lunarlander/simple_{simple_type}_ei_{ei_type}/critic_{critic_lr}_actor_{actor_lr}/{configuration}.pkl'

    buffer_path = info_path.replace('.pkl', f'_buffer{seed}.pkl')
    buffer.load(buffer_path)
    # print(buffer.episode_rewards.mean())
    n_action = int(buffer.action.max()+1)
    feature_dim = buffer.state.shape[-1] + n_action
    buffer_state = buffer.state[:buffer.ptr]
    buffer_action = buffer.action[:buffer.ptr]
    buffer_done = buffer.done[:buffer.ptr]
    if network_name == 'critic':
        buffer_hidden = buffer.hidden_state[:buffer.ptr, 0]
    elif network_name == 'actor':
        buffer_hidden = buffer.hidden_state[:buffer.ptr, 1]

    done = np.asarray(buffer_done).reshape(-1).astype(bool)
    # Build lengths: split at done==1
    ends = np.where(done)[0]
    # Assume done marks the last step of an episode
    hmm_lengths = []
    start = 0
    for e in ends:
        L = (e - start + 1)
        if L > 1:
            hmm_lengths.append(L)
        start = e + 1
    if start < len(done):  # trailing partial episode
        hmm_lengths.append(len(done) - start)

    ## split train and test and calculate the score
    zero_plus_boundaries = np.cumsum(hmm_lengths)
    train_indices = []
    test_indices = []
    start = 0
    for i, end in enumerate(zero_plus_boundaries):
        L = end - start
        if i % 5 == 0:
            test_indices.append((start, end))
        else:
            train_indices.append((start, end))
        start = end
    # Calculate train and test score
    train_buffer_hidden = np.concatenate([buffer_hidden[s:e] for s,e in train_indices], axis=0)
    test_buffer_hidden = np.concatenate([buffer_hidden[s:e] for s,e in test_indices], axis=0)
    train_buffer_done = np.concatenate([buffer_done[s:e] for s,e in train_indices], axis=0)
    test_buffer_done = np.concatenate([buffer_done[s:e] for s,e in test_indices], axis=0)

    # Fit on train
    train_labels, train_post, train_model_hmm, train_scaler, train_lengths = fit_hmm_gaussian(
        train_buffer_hidden,
        train_buffer_done,
        n_components=n_components,   # or n_components depending on your function signature
        cov_type="tied",
        n_iter=200,
        seed=0
    )

    # Prepare features
    X_train = train_scaler.transform(train_buffer_hidden)
    X_test  = train_scaler.transform(test_buffer_hidden)

    # Length lists
    test_lengths = [e - s for s, e in test_indices]

    # Sum log-likelihoods (hmmlearn default)
    train_ll_sum = train_model_hmm.score(X_train, lengths=train_lengths)
    test_ll_sum, test_posterior = train_model_hmm.score_samples(X_test,  lengths=test_lengths)

    # Convert to average per timestep
    train_T = int(np.sum(train_lengths))
    test_T  = int(np.sum(test_lengths))

    train_ll_avg = train_ll_sum / train_T
    test_ll_avg  = test_ll_sum / test_T

    train_labels_hmm = train_labels
    test_labels_hmm = train_model_hmm.predict(X_test, lengths=test_lengths)


    train_confidence = train_post.max(1).mean()
    test_confidence = test_posterior.max(1).mean()
    train_switch_rate = np.where(np.diff(train_labels_hmm),1,0).mean()
    test_switch_rate = np.where(np.diff(test_labels_hmm),1,0).mean()
    transition = train_model_hmm.transmat_    # shape [K, K]

    stickiness = np.trace(transition)/transition.shape[0]
    entropy = -np.mean((transition * np.log(transition+1e-6))) * transition.shape[1]

    buffer_phase = tag_lunar_lander_phases_batch_2(buffer_state, buffer_action, buffer_done)  # [L,]
    train_state = np.concatenate([buffer_state[s:e] for s,e in train_indices], axis=0)
    test_state = np.concatenate([buffer_state[s:e] for s,e in test_indices], axis=0)
    train_action = np.concatenate([buffer_action[s:e] for s,e in train_indices], axis=0)
    test_action = np.concatenate([buffer_action[s:e] for s,e in test_indices], axis=0)
    train_phase = np.concatenate([buffer_phase[s:e] for s,e in train_indices], axis=0)
    test_phase = np.concatenate([buffer_phase[s:e] for s,e in test_indices], axis=0)

    n_phase = buffer_phase.max()+1

    train_mi, train_nmi = mutual_information(train_phase, train_labels_hmm)
    test_mi, test_nmi = mutual_information(test_phase, test_labels_hmm)
    train_ami = adjusted_mutual_info_score(train_phase, train_labels_hmm)
    test_ami = adjusted_mutual_info_score(test_phase, test_labels_hmm)

    if record:
        with open('./rl/hmm_metric.txt', 'a+') as f:
            f.write(configuration+' '+network_name+'\n')
            f.write('HMM Train ll: '+str(train_ll_avg)+'\n')
            f.write('HMM Test ll: '+str(test_ll_avg)+'\n')
            f.write('HMM Train confidence: '+str(train_confidence)+'\n')
            f.write('HMM Test confidence: '+str(test_confidence)+'\n')
            f.write('HMM Train switch rate: '+str(train_switch_rate)+'\n')
            f.write('HMM Test switch rate: '+str(test_switch_rate)+'\n')
            f.write('HMM T stickiness: '+str(stickiness)+'\n')
            f.write('HMM T entropy: '+str(entropy)+'\n')
            f.write('Train NMI: '+str(train_nmi)+'\n')
            f.write('Test NMI: '+str(test_nmi)+'\n')
    return train_ll_avg, test_ll_avg, train_switch_rate, test_switch_rate, stickiness, entropy, train_nmi, test_nmi
