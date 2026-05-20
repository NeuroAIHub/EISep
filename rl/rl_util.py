import numpy as np
import os
from matplotlib import pyplot as plt

def tag_lunar_lander_phases_batch(
    buffer_state: np.ndarray,   # [L, state_dim]
    buffer_action: np.ndarray,  # [L, 1] or [L]
    buffer_done: np.ndarray,    # [L, 1] or [L]
):
    """
    Tag each timestep with a behavioral phase in LunarLander using NumPy.

    Phases:
        0: Free fall / early descent
        1: Braking / vertical control (main engine)
        2: Horizontal alignment / attitude control
        3: Touchdown preparation
        4: Landed
        5: Crash

    Args:
        buffer_state: [L, state_dim], columns = [x, y, vx, vy, angle, ang_vel, legL, legR]
        buffer_action: [L, 1] or [L], integers {0,1,2,3}
        buffer_done: [L, 1] or [L], bool or 0/1

    Returns:
        phases: np.ndarray (int64) with shape [L], values ∈ {0..5}.
    """
    # ---- Extract state ----
    x  = buffer_state[:, 0]
    y  = buffer_state[:, 1]
    vx = buffer_state[:, 2]
    vy = buffer_state[:, 3]
    a  = buffer_state[:, 4]  # angle
    va = buffer_state[:, 5]  # angular velocity

    # Leg contact (assume 0/1 floats)
    legL = buffer_state[:, 6] if buffer_state.shape[1] > 6 else np.zeros_like(x)
    legR = buffer_state[:, 7] if buffer_state.shape[1] > 7 else np.zeros_like(x)
    contact = (legL > 0.5) | (legR > 0.5)

    # ---- Action & done ----
    action = buffer_action.reshape(-1).astype(int)
    done = buffer_done.reshape(-1).astype(bool)

    # ---- Precompute thresholds from your statistics ----
    high_alt    = y > 0.80
    mid_alt     = (y <= 0.80) & (y > 0.30)
    low_alt     = (y <= 0.30) & (y > 0.05)
    near_ground = y <= 0.05

    fast_down   = vy < -0.30
    slow_vertical = np.abs(vy) < 0.15
    very_fast_v = np.abs(vy) > 0.40

    far_x       = np.abs(x) > 0.25
    very_far_x  = np.abs(x) > 0.40

    large_vx    = np.abs(vx) >= 0.30

    small_tilt     = np.abs(a) < 0.15
    moderate_tilt  = (np.abs(a) >= 0.15) & (np.abs(a) < 0.25)
    large_tilt     = np.abs(a) >= 0.25
    very_large_tilt = np.abs(a) > 0.35

    main_engine = action == 2

    # ---- Initialize all phases = 0 (free fall default) ----
    L = buffer_state.shape[0]
    phases = np.zeros(L, dtype=np.int64)

    # ---- Braking (1) ----
    braking_mask = (
        (high_alt | mid_alt) &
        main_engine &
        (~fast_down | slow_vertical)
    )
    phases[braking_mask] = 1

    # ---- Horizontal alignment (2) ----
    horiz_mask = (
        (mid_alt | low_alt) &
        (far_x | large_vx | large_tilt)
    )
    phases[horiz_mask] = 2

    # ---- Touchdown preparation (3) ----
    prep_mask = (
        (~contact) &
        (low_alt | near_ground) &
        (np.abs(vy) < 0.30) &
        (np.abs(a) < 0.25)
    )
    phases[prep_mask] = 3

    # ---- Landed (4) ----
    landed_mask = (
        contact &
        (y <= 0.30) &
        slow_vertical &
        (np.abs(a) < 0.20)
    )
    phases[landed_mask] = 4

    # ---- Crash (5) ----
    crash_mask = (
        done &
        (~contact) &
        (
            near_ground |
            very_large_tilt |
            very_fast_v |
            (np.abs(vx) > 0.50) |
            very_far_x
        )
    )
    phases[crash_mask] = 5

    # ---- Explicit free-fall (0): optional check
    # free_fall_mask = high_alt & fast_down & (~main_engine)
    # phases[free_fall_mask] = 0

    return phases


def tag_lunar_lander_phases_batch_2(
    buffer_state: np.ndarray,   # [L, 8], [x,y,vx,vy,angle,ang_vel,legL,legR]
    buffer_action: np.ndarray,  # [L,] or [L,1], int {0..3}
    buffer_done: np.ndarray,    # [L,] or [L,1], bool or 0/1
) -> np.ndarray:
    """
    Phase labels with fixed, env-based thresholds (LunarLander-v3).

    Phases:
        0: Free fall / early descent
        1: Braking / vertical control
        2: Horizontal alignment / attitude control
        3: Touchdown preparation
        4: Landed / stable contact
        5: Crash (terminal failure)
    """
    x  = buffer_state[:, 0]
    y  = buffer_state[:, 1]
    vx = buffer_state[:, 2]
    vy = buffer_state[:, 3]
    theta = buffer_state[:, 4]
    ang_vel = buffer_state[:, 5]

    legL = buffer_state[:, 6]
    legR = buffer_state[:, 7]
    contact = (legL > 0.5) | (legR > 0.5)

    action = buffer_action.reshape(-1).astype(int)
    done   = buffer_done.reshape(-1).astype(bool)

    L = buffer_state.shape[0]
    phases = np.zeros(L, dtype=np.int64)

    categorized_mask = np.zeros(L, dtype=bool)
    # Convenience masks
    high_alt   = y > 1.0
    mid_alt    = (y > 0.4) & (y <= 1.0)
    low_alt    = (y > 0.1) & (y <= 0.4)
    near_ground = y <= 0.1

    fast_down  = vy < -2.0
    near_zero_vy = np.abs(vy) < 0.5
    moderate_safe_vy = np.abs(vy) < 1.0
    very_fast_vy = np.abs(vy) > 3.0

    small_vx   = np.abs(vx) < 0.5
    large_vx   = np.abs(vx) >= 2.0

    near_pad   = np.abs(x) <= 0.3
    far_x      = np.abs(x) > 1.0

    small_tilt = np.abs(theta) < 0.2
    medium_tilt = (np.abs(theta) >= 0.2) & (np.abs(theta) < 0.4)
    large_tilt = np.abs(theta) >= 0.4
    very_large_tilt = np.abs(theta) > 0.7

    small_ang_vel = np.abs(ang_vel) < 1.0
    large_ang_vel = np.abs(ang_vel) >= 2.0

    main_engine = (action == 2)

    # Phase 1: Braking
    braking_mask = (y > 0.0) & main_engine & (vy > -3.0) & (~contact)
    phases[braking_mask] = 1

    # Phase 2: Horizontal alignment / attitude control
    horiz_mask = (
        (np.abs(vx) >= 0.5) |
        (np.abs(theta) >= 0.3) |
        (np.abs(ang_vel) >= 0.15)
    ) & (~contact)
    phases[horiz_mask] = 2

    # Phase 3: Touchdown preparation
    prep_mask = (
        (~contact) &
        (y <= 0.2) &
        moderate_safe_vy &
        (np.abs(theta) < 0.3)
    )
    phases[prep_mask] = 3

    # Phase 4: Landed / stable contact
    landed_mask = (
        contact &
        (y <= 0.3) &
        near_zero_vy &
        (~very_large_tilt) &
        (small_vx)
    )
    phases[landed_mask] = 4

    # Phase 5: Crash
    crash_mask = (
        done &
        (~contact) &
        (
            near_ground |
            very_large_tilt |
            very_fast_vy |
            (np.abs(vx) > 3.0) |
            far_x
        )
    )
    phases[crash_mask] = 5

    # Phase 0: Free fall (default), but we can explicitly tag:
    free_fall_mask = high_alt & fast_down & (~main_engine)
    phases[free_fall_mask] = 0  # they're already 0, but this documents intent
    categorized_mask |= (braking_mask | horiz_mask | prep_mask | landed_mask | crash_mask | free_fall_mask)
    uncategorized_frac = 1.0 - categorized_mask.mean()

    return phases, uncategorized_frac


def event_selectivity(buffer_hidden: np.ndarray,
                      event_mask: np.ndarray,
                      eps: float = 1e-6) -> np.ndarray:
    """
    Compute event-specific selectivity for each neuron.

    Args:
        buffer_hidden: [L, hidden_size]
        event_mask: [L,] bool array, True where event occurs.
        eps: small constant for numerical stability.

    Returns:
        selectivity: [hidden_size,], event vs non-event.
    """
    event_mask = event_mask.astype(bool)
    pos_h = buffer_hidden[event_mask]      # [n_event, H]
    neg_h = buffer_hidden[~event_mask]     # [n_non, H]

    pos_mean = pos_h.mean(axis=0)
    neg_mean = neg_h.mean(axis=0)
    pos_std  = pos_h.std(axis=0)
    neg_std  = neg_h.std(axis=0)

    sel = (pos_mean - neg_mean) / (pos_std + neg_std + eps)
    return sel


def subspace_alignment(vectors_a, vectors_b, d=10):
    ### vectors_a: [N1, H]
    ### vectors_b: [N2, H]
    u1, s1, vh1 = np.linalg.svd(vectors_a.T, full_matrices=False)
    u2, s2, vh2 = np.linalg.svd(vectors_b.T, full_matrices=False)
    pcs_a = u1[:, :d]  # [H, d]
    pcs_b = u2[:, :d]  # [H, d]
    product = pcs_a.T @ pcs_b
    u, s, vh = np.linalg.svd(product)
    return s

def subspace_alignments_across_index(all_hidden, specific_idxs):
    n_idx = len(specific_idxs)
    alignments = np.zeros((n_idx, n_idx))
    for i in range(n_idx):
        for j in range(i, n_idx):
            s = subspace_alignment(
                all_hidden[specific_idxs[i]==1],
                all_hidden[specific_idxs[j]==1],
            )
            print(f'Alignment between action {i} and action {j}:', s)
            alignments[i,j] = s.mean()
            alignments[j,i] = s.mean()
    return alignments

def sep_corr(Z):  # Z: [T, r]
    C = np.corrcoef(Z, rowvar=False)  # [r, r]
    r = C.shape[0]
    off = np.abs(C - np.eye(r))
    return 1.0 - off.sum() / (r*(r-1))

def sep_phase_fisher(Z, phase, eps=1e-6):
    # Z: [T, r], phase: [T] int in {0..P-1}
    P = int(phase.max()) + 1
    r = Z.shape[1]
    mu = Z.mean(axis=0)
    Sb = np.zeros((r, r))
    Sw = np.zeros((r, r))
    idxs_count = np.zeros(P, dtype=int)
    for p in range(P):
        idx = (phase == p)
        idxs_count[p] = idx.sum()
        if idx.sum() < 5:
            continue
        Zp = Z[idx]
        mup = Zp.mean(axis=0)
        Cp = np.cov(Zp, rowvar=False) if Zp.shape[0] > 1 else np.zeros((r,r))
        Sb += idx.sum() * np.outer(mup - mu, mup - mu)
        Sw += (idx.sum() - 1) * Cp
    return np.trace(Sb) / (np.trace(Sw) + eps), idxs_count


def mutual_information(label_1, label_2, eps=1e-12, normalized=True):
    """
    phase: [T] int (0..P-1)
    state: [T] int (0..K-1)
    Returns:
        MI (nats), and optionally NMI (0..1)
    """
    label_1 = np.asarray(label_1).astype(int).ravel()
    label_2 = np.asarray(label_2).astype(int).ravel()
    assert label_1.shape == label_2.shape

    P = label_1.max() + 1
    K = label_2.max() + 1

    # joint counts
    joint = np.zeros((P, K), dtype=np.float64)
    np.add.at(joint, (label_1, label_2), 1.0)

    T = joint.sum()
    p_ps = joint / (T + eps)
    p_p  = p_ps.sum(axis=1, keepdims=True)  # [P,1]
    p_s  = p_ps.sum(axis=0, keepdims=True)  # [1,K]

    # MI = sum p_ps * log(p_ps/(p_p p_s))
    ratio = (p_ps + eps) / (p_p * p_s + eps)
    mi = np.sum(p_ps * np.log(ratio))

    if not normalized:
        return mi

    Hp = -np.sum(p_p * np.log(p_p + eps))
    Hs = -np.sum(p_s * np.log(p_s + eps))
    nmi = mi / (np.sqrt(Hp * Hs) + eps)
    return mi, nmi

def savefig(figpath, dpi=400):
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    plt.savefig(figpath, dpi=dpi)
    print(figpath)
