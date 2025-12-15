import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import pandas as pd
# =========================================
# 基础：从 W 拆成 A+ / A-
# =========================================
def build_signed_digraph_from_W(W, threshold=None, keep_diag=False):
    """
    返回 (G_plus, G_minus) 两个 DiGraph；边权非负。
    A_plus[u,v] = max(W[u,v],0)；A_minus[u,v] = max(-W[u,v],0)
    """
    W_pos = np.maximum(W, 0.0).copy()
    W_neg = np.maximum(-W, 0.0).copy()

    if not keep_diag:
        np.fill_diagonal(W_pos, 0.0)
        np.fill_diagonal(W_neg, 0.0)

    if threshold is not None:
        W_pos = np.where(W_pos >= threshold, W_pos, 0.0)
        W_neg = np.where(W_neg >= threshold, W_neg, 0.0)

    Gp = nx.from_numpy_array(W_pos, create_using=nx.DiGraph)
    Gn = nx.from_numpy_array(W_neg, create_using=nx.DiGraph)
    for u, v, d in Gp.edges(data=True):
        d['weight'] = float(W_pos[u, v])
    for u, v, d in Gn.edges(data=True):
        d['weight'] = float(W_neg[u, v])
    return Gp, Gn

def digraph_to_adj(G):
    n = G.number_of_nodes()
    A = np.zeros((n, n), dtype=float)
    for u, v, d in G.edges(data=True):
        A[u, v] += d.get('weight', 1.0)
    return A

# =========================================
# A. 2n 提升（lifting）
# -----------------------------------------
# 对有向符号图，构造 2n×2n 的非负有向邻接：
#   B = [[A+, A-],
#        [A-, A+]]
# 直观：正边连接“同符号副本”，负边连接“异符号副本”
# 然后对 B 跑 MCL / 谱分割，最后把 2n 个标签还原到 n 个节点
# =========================================
def lift_signed_digraph(A_pos, A_neg):
    A_pos = np.asarray(A_pos); A_neg = np.asarray(A_neg)
    n = A_pos.shape[0]
    top = np.concatenate([A_pos, A_neg], axis=1)
    bot = np.concatenate([A_neg, A_pos], axis=1)
    B = np.concatenate([top, bot], axis=0)
    return B  # (2n, 2n)

def row_stochastic(M):
    rs = M.sum(axis=1, keepdims=True) + 1e-12
    return M / rs

def mcl_matrix(M, expand_power=2, inflate_r=1.6, self_loop=1.0, max_iter=200, tol=1e-9, verbose=False):
    # 在行随机游走空间做 MCL
    M = M.copy().astype(float)
    M = M + np.eye(M.shape[0]) * self_loop
    M = row_stochastic(M)
    last = None
    for it in range(max_iter):
        M = np.linalg.matrix_power(M, expand_power)
        M = np.power(M, inflate_r)
        M = row_stochastic(M)
        if last is not None:
            diff = np.abs(M - last).sum()
            if verbose:
                print(f"[MCL-lift] iter={it} diff={diff:.3e}")
            if diff < tol:
                break
        last = M.copy()
    return M

def mcl_classic(M, expand_power=2, inflate_r=2.2, self_loop=0.1,
                prune=1e-3, max_iter=200, tol=1e-7, verbose=False):
    """
    Classic MCL in column-stochastic space.
    Parameters mirror mcl_matrix; adds `prune`.
      - expand_power: 2 or 3 are common
      - inflate_r:    1.8~2.5 typical (2.0~2.2 often good)
      - self_loop:    small positive; scaled by matrix mean
      - prune:        zero out small entries after inflation
    Returns:
      Column-stochastic matrix after convergence.
    """
    M = np.asarray(M, dtype=float).copy()
    n = M.shape[0]

    # add self-loop scaled by mean (avoid overly lazy walk)
    mu = M.mean() if M.size else 0.0
    M = M + np.eye(n) * (self_loop * (mu if mu > 0 else 1.0))

    def col_stochastic(X):
        cs = X.sum(axis=0, keepdims=True) + 1e-12
        return X / cs

    # initialize as column-stochastic
    M = col_stochastic(M)

    last = None
    for it in range(max_iter):
        # expansion
        if expand_power == 2:
            M = M @ M
        else:
            M = np.linalg.matrix_power(M, expand_power)

        # inflation
        M = np.power(M, inflate_r)

        # pruning (sparsify) + renormalize
        if prune is not None and prune > 0.0:
            M[M < prune] = 0.0
        M = col_stochastic(M)

        if last is not None:
            diff = np.abs(M - last).sum()
            if verbose:
                print(f"[MCL-classic] iter={it} diff={diff:.3e}")
            if diff < tol:
                break
        last = M.copy()

    return M

def labels_from_mcl_matrix(M):
    raw = M.argmax(axis=1)
    uniq = {v:i for i, v in enumerate(np.unique(raw))}
    return np.array([uniq[v] for v in raw], dtype=int)

def signed_partition_lifting_MCL(W, threshold=None, keep_diag=True,
                                 expand_power=2, inflate_r=1.6, self_loop=1.0):
    Gp, Gn = build_signed_digraph_from_W(W, threshold=threshold, keep_diag=keep_diag)
    Apos, Aneg = digraph_to_adj(Gp), digraph_to_adj(Gn)
    B = lift_signed_digraph(Apos, Aneg)  # (2n, 2n)
    M = mcl_classic(B, expand_power=expand_power, inflate_r=inflate_r, self_loop=self_loop)
    labels_2n = labels_from_mcl_matrix(M)
    # 折回 n 个节点：取两个副本的多数（若不一致则取第一个）
    n = Apos.shape[0]
    lab_top, lab_bot = labels_2n[:n], labels_2n[n:]
    labels = np.where(lab_top == lab_bot, lab_top, lab_top)  # 简单聚合；也可加一致性修复
    return (Gp, Gn), labels

def signed_partition_lifting_spectral(W, k, threshold=None, keep_diag=True, random_state=0):
    Gp, Gn = build_signed_digraph_from_W(W, threshold=threshold, keep_diag=keep_diag)
    Apos, Aneg = digraph_to_adj(Gp), digraph_to_adj(Gn)
    B = lift_signed_digraph(Apos, Aneg)  # (2n, 2n)
    # 取 B 的前 k 个奇异向量（或特征向量），再 k-means
    try:
        U, S, Vt = svds(csr_matrix(B), k=min(k, B.shape[0]-1))
        X = U
    except Exception:
        U, S, Vt = np.linalg.svd(B, full_matrices=False)
        X = U[:, :k]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
    labels_2n = km.labels_
    n = Apos.shape[0]
    lab_top, lab_bot = labels_2n[:n], labels_2n[n:]
    labels = np.where(lab_top == lab_bot, lab_top, lab_top)
    return (Gp, Gn), labels

def directed_signed_spectral_from_W(W, k, threshold=None, keep_diag=False, random_state=0):
    Gp, Gn = build_signed_digraph_from_W(W, threshold=threshold, keep_diag=keep_diag)
    Apos, Aneg = digraph_to_adj(Gp), digraph_to_adj(Gn)
    kout_p, kin_p = Apos.sum(axis=1), Apos.sum(axis=0)
    kout_n, kin_n = Aneg.sum(axis=1), Aneg.sum(axis=0)

    def norm_embed(A, kout, kin, k_use):
        kout = np.maximum(kout, 1e-12)
        kin  = np.maximum(kin,  1e-12)
        Dout_is = np.diag(1.0/np.sqrt(kout))
        Din_is  = np.diag(1.0/np.sqrt(kin))
        M = Dout_is @ A @ Din_is
        try:
            U, S, Vt = svds(csr_matrix(M), k=min(k_use, M.shape[0]-1))
            X = U
        except Exception:
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            X = U[:, :k_use]
        return X

    k_half = max(1, k//2)
    Xp = norm_embed(Apos, kout_p, kin_p, k_half)   # 吸引
    Xn = norm_embed(Aneg, kout_n, kin_n, k - k_half)  # 排斥
    # 负边用相反号（把“相似度”翻成“差异度”）
    X = np.concatenate([Xp, -Xn], axis=1)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
    labels = km.labels_
    return (Gp, Gn), labels
def signed_directed_modularity(W, labels):
    """
    有向符号模块度（Traag & Bruggeman 风格的有向扩展）：
    Q =  (1/m+) * sum_{ij in same} [A+_ij - kout+_i * kin+_j / m+]
       - (1/m-) * sum_{ij in same} [A-_ij - kout-_i * kin-_j / m-]
    正边鼓励同簇，负边惩罚同簇（鼓励跨簇）。
    """
    Gp, Gn = build_signed_digraph_from_W(W, keep_diag=False)
    Apos, Aneg = digraph_to_adj(Gp), digraph_to_adj(Gn)
    kout_p, kin_p = Apos.sum(axis=1), Apos.sum(axis=0)
    kout_n, kin_n = Aneg.sum(axis=1), Aneg.sum(axis=0)
    mp = Apos.sum(); mn = Aneg.sum()
    labels = np.asarray(labels)

    def part_Q(A, kout, kin, m, sign=+1.0):
        if m <= 0: return 0.0
        Q = 0.0
        for i in range(A.shape[0]):
            li = labels[i]
            for j in range(A.shape[1]):
                if labels[j] == li:
                    Pij = (kout[i] * kin[j]) / (m + 1e-12)
                    Q += (A[i, j] - Pij)
        return sign * (Q / (m + 1e-12))

    return float(part_Q(Apos, kout_p, kin_p, mp, +1.0) + part_Q(Aneg, kout_n, kin_n, mn, -1.0))

def signed_frustration(W, labels):
    """
    挫折率（越小越好）：
      Frust = ( 正边跨簇 + 负边同簇 ) / ( |A+| + |A-| )
    """
    Gp, Gn = build_signed_digraph_from_W(W, keep_diag=False)
    Apos, Aneg = digraph_to_adj(Gp), digraph_to_adj(Gn)
    labels = np.asarray(labels)
    mp, mn = Apos.sum(), Aneg.sum()
    if mp + mn <= 0: return 0.0

    # 正边跨簇
    cross_pos = 0.0
    same_neg  = 0.0
    n = Apos.shape[0]
    for i in range(n):
        for j in range(n):
            if labels[i] != labels[j]:
                cross_pos += Apos[i, j]
            else:
                same_neg  += Aneg[i, j]
    return float((cross_pos + same_neg) / (mp + mn + 1e-12))

def signed_weighted_ncut(W, labels):
    """
    有向符号 Ncut（越小越好）：
    对每个簇 C：
      Ncut_C = [ (A+_{C,~C} + A-_{C,C}) / vol_out^+(C∪-)  +  (A+_{~C,C} + A-_{C,C}) / vol_in^+(C∪-) ] / 2
    直觉：把“好”的流量视作 正边留在簇内 + 负边跨簇；“坏”的流量反过来。
    实现上给一个合理但简洁的近似。
    """
    Gp, Gn = build_signed_digraph_from_W(W, keep_diag=False)
    Apos, Aneg = digraph_to_adj(Gp), digraph_to_adj(Gn)
    kout_p, kin_p = Apos.sum(axis=1), Apos.sum(axis=0)
    kout_n, kin_n = Aneg.sum(axis=1), Aneg.sum(axis=0)
    labels = np.asarray(labels)
    K = np.unique(labels)
    ncut = 0.0
    for c in K:
        C = np.where(labels == c)[0]
        notC = np.where(labels != c)[0]
        # “坏”流：正边出簇、正边入簇、负边留在簇内
        bad_out = Apos[np.ix_(C, notC)].sum() + Aneg[np.ix_(C, C)].sum()
        bad_in  = Apos[np.ix_(notC, C)].sum() + Aneg[np.ix_(C, C)].sum()
        vol_out = (kout_p[C].sum() + kout_n[C].sum()) + 1e-12
        vol_in  = (kin_p[C].sum()  + kin_n[C].sum()) + 1e-12
        ncut += 0.5 * (bad_out / vol_out + bad_in / vol_in)
    return float(ncut / len(K))


# —— 小工具：把标签压缩为 0..K-1，并返回有效簇数 ——
def _compress_labels(labels):
    labels = np.asarray(labels)
    uniq = {v:i for i, v in enumerate(np.unique(labels))}
    lab  = np.array([uniq[v] for v in labels], dtype=int)
    return lab, len(uniq)
# —— 单次评估：给定 W 与 k，跑一种方法求分数 ——
def _eval_one_k(W, k, method, keep_diag=True, random_state=0):
    if method == "lift":
        (Gp, Gn), lab = signed_partition_lifting_spectral(W, k=k, keep_diag=keep_diag, random_state=random_state)
    elif method == "direct":
        (Gp, Gn), lab = directed_signed_spectral_from_W(W, k=k, keep_diag=keep_diag, random_state=random_state)
    else:
        raise ValueError("method must be 'lift' or 'direct'")
    lab, k_eff = _compress_labels(lab)
    Q  = signed_directed_modularity(W, lab)
    Fr = signed_frustration(W, lab)
    Nc = signed_weighted_ncut(W, lab)
    return {"k": k, "k_eff": k_eff, "Q": Q, "Frustr": Fr, "Ncut": Nc, "labels": lab}

# —— 多次重启以稳健（可选）：对同一 k 重复多次取最好/平均 ——
def _eval_k_with_restarts(W, k, method, keep_diag, restarts=3, seed=0, agg="bestQ"):
    rows = []
    for r in range(restarts):
        row = _eval_one_k(W, k, method, keep_diag=keep_diag, random_state=seed + r)
        rows.append(row)
    # 聚合：默认取 Q 最大的那次（也可以改成平均）
    if agg == "bestQ":
        best = max(rows, key=lambda d: (d["Q"], -d["Frustr"], -d["Ncut"]))
        return best
    if agg == "bestFrustr":
        best = min(rows, key=lambda d: (-d["Frustr"], d["Ncut"], -d["Q"]))
        return best
    elif agg == "mean":
        out = {k: np.mean([d[k] for d in rows]) for k in rows[0].keys() if k not in ("k",)}
        out["k"] = k
        out["k_eff"] = int(round(out["k_eff"]))
        return out
    else:
        raise ValueError("agg must be 'bestQ' or 'mean'")

def scan_k_signed_spectral(W, method="lift", k_min=2, k_max=10, keep_diag=True,
                           restarts=3, seed=0, agg="bestFrustr",
                           q_threshold=0.99, require_keff=True):
    """
    method: 'lift' or 'direct'
    restarts: 每个 k 的随机重启次数
    agg: 'bestQ' 取 Q 最佳那次；或 'mean' 对重启取均值
    q_threshold: “保守选择”阈值；选 Q ≥ q_threshold * Q_max 的最小 k
    require_keff: 若 True，仅在 k_eff==k 才参与保守选择（避免塌缩）
    返回: (df, picks)：
      df: 每个 k 的得分表
      picks: {'best_Q':k, 'best_Frustr':k, 'best_Ncut':k, 'conservative':k}
    """
    k_vals = list(range(k_min, min(k_max, W.shape[0]-1) + 1))
    rows = []
    for k in k_vals:
        rows.append(_eval_k_with_restarts(W, k, method, keep_diag, restarts, seed, agg))
    df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    # 三种直接准则
    # 当出现并列，用次级准则打破平局
    k_best_Q   = df.sort_values(["Q", "Frustr", "Ncut"], ascending=[False, True, True]).iloc[0]["k"]
    k_best_Fr  = df.sort_values(["Frustr", "Ncut", "Q"], ascending=[True, True, False]).iloc[0]["k"]
    k_best_Nc  = df.sort_values(["Ncut", "Frustr", "Q"], ascending=[True, True, False]).iloc[0]["k"]

    # 保守选择：Q 接近最优的最小 k（可选要求 k_eff==k）
    Qmax = df["Q"].max()
    eligible = df[df["Q"] >= q_threshold * Qmax]
    if require_keff:
        eligible = eligible[eligible["k"] == eligible["k_eff"]]
    if len(eligible) == 0:
        k_cons = int(k_best_Q)  # 回退
    else:
        # 在候选里选 k 最小；若有多个，再选 Frustr/Ncut 更小的
        k_cons = int(eligible.sort_values(["k", "Frustr", "Ncut"], ascending=[True, True, True]).iloc[0]["k"])

    picks = {"best_Q": int(k_best_Q), "best_Frustr": int(k_best_Fr), "best_Ncut": int(k_best_Nc), "conservative": k_cons}
    return df, picks

def signed_spectral_W(W, k_min=2, k_max=10, restarts=5, seed=0):
    df_dir,   picks_dir   = scan_k_signed_spectral(
        W,  method="direct", k_min=k_min, k_max=k_max,
        keep_diag=False, restarts=restarts, seed=seed,
        agg="bestQ", q_threshold=0.99, require_keff=True
    )
    return df_dir, picks_dir
