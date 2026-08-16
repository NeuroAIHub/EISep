"""Run the Yang19 EISep forward-sign stability experiment."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import neurogym as ngym
import numpy as np
import torch
from neurogym.envs import get_collection
from neurogym.utils import RandomSchedule
from neurogym.wrappers import ScheduleEnvs
from torch import nn

from net.common import init_model


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_SEEDS = 5
EPOCHS = 3000
HIDDEN_SIZE = 64
BATCH_SIZE = 32
SEQ_LEN = 100
LEARNING_RATE = 1e-2
EVAL_TRIALS = 100
EARLY_FRACTION = 0.10
DT = 100
DIM_RING = 2


def make_environment():
    kwargs = {"dt": DT, "dim_ring": DIM_RING}
    tasks = get_collection("yang19")
    envs = [ngym.make(task, **kwargs) for task in tasks]
    return ScheduleEnvs(
        envs,
        schedule=RandomSchedule(len(envs)),
        env_input=True,
    )


def flip_metrics(sampled, preferred):
    """Compute forward-sample and learned-preference sign stability metrics."""
    # sampled: (update, time, neuron); preferred: (update, neuron)
    n_update, n_time, _ = sampled.shape
    window = max(1, int(round(n_update * EARLY_FRACTION)))

    within = sampled[:, 1:] != sampled[:, :-1]
    between = sampled[1:, 0] != sampled[:-1, -1]
    preferred_changes = preferred[1:] != preferred[:-1]
    disagreement = sampled != preferred[:, None, :]

    transition_count = within.sum(axis=(1, 2)).astype(np.int64)
    transition_denom = (n_time - 1) * sampled.shape[-1]
    per_update_sample_flip_rate = transition_count / transition_denom
    per_update_disagreement_rate = disagreement.mean(axis=(1, 2))
    per_update_preferred_flip_rate = np.r_[
        0.0, preferred_changes.mean(axis=1)
    ]

    ever_changed = preferred_changes.any(axis=0)
    flips_per_neuron = preferred_changes.sum(axis=0)
    last_flip = np.full(preferred.shape[1], -1, dtype=np.int64)
    for neuron in range(preferred.shape[1]):
        indices = np.flatnonzero(preferred_changes[:, neuron])
        if indices.size:
            # A change between rows i and i+1 becomes visible at update i+1.
            last_flip[neuron] = indices[-1] + 1

    total_adjacent_flips = within.sum() + between.sum()
    total_adjacent_transitions = (
        within.size + between.size
    )
    return {
        "sample_flip_rate_all": float(total_adjacent_flips / total_adjacent_transitions),
        "sample_flip_rate_early_10pct": float(
            per_update_sample_flip_rate[:window].mean()
        ),
        "sample_flip_rate_late_10pct": float(
            per_update_sample_flip_rate[-window:].mean()
        ),
        "sample_disagrees_with_preferred_all": float(disagreement.mean()),
        "sample_disagrees_with_preferred_early_10pct": float(
            per_update_disagreement_rate[:window].mean()
        ),
        "sample_disagrees_with_preferred_late_10pct": float(
            per_update_disagreement_rate[-window:].mean()
        ),
        "preferred_flip_rate_per_update_neuron": float(preferred_changes.mean()),
        "preferred_flip_rate_early_10pct": float(
            preferred_changes[: max(1, window - 1)].mean()
        ),
        "preferred_flip_rate_late_10pct": float(
            preferred_changes[-window:].mean()
        ),
        "neurons_ever_changing_preferred_sign": float(ever_changed.mean()),
        "preferred_flips_per_neuron_mean": float(flips_per_neuron.mean()),
        "preferred_flips_per_neuron_median": float(np.median(flips_per_neuron)),
        "preferred_flips_per_neuron_max": int(flips_per_neuron.max()),
        "final_sign_established_by_10pct": float((last_flip < window).mean()),
        "final_sign_established_by_25pct": float(
            (last_flip < int(round(n_update * 0.25))).mean()
        ),
        "final_sign_established_by_50pct": float(
            (last_flip < int(round(n_update * 0.50))).mean()
        ),
        "per_update_sample_flip_rate": per_update_sample_flip_rate,
        "per_update_disagreement_rate": per_update_disagreement_rate,
        "per_update_preferred_flip_rate": per_update_preferred_flip_rate,
        "last_preferred_flip_update": last_flip,
        "preferred_flips_per_neuron": flips_per_neuron,
    }


def train_one_seed(seed, env):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    env.reset(seed=seed)

    dataset = ngym.Dataset(
        env,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
    )
    dataset.seed(seed)
    dataset._cache()
    ob_size = env.observation_space.shape[0]
    act_size = env.action_space.n
    net = init_model(ob_size, HIDDEN_SIZE, act_size, "eisep").to(DEVICE)
    net.train()
    net.lstm.enable_sign_tracking()

    class_weights = torch.tensor(
        [0.05] + [1.0] * (act_size - 1),
        device=DEVICE,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0,
    )
    loss_tensors = []

    started = time.time()
    for update in range(EPOCHS):
        inputs, targets = dataset()
        inputs = torch.from_numpy(inputs).float().to(DEVICE)
        targets = torch.from_numpy(targets).long().to(DEVICE)

        outputs, _, _ = net(inputs)
        decoding_loss = criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            targets.reshape(-1),
        )
        total_loss = decoding_loss + net.get_ei_entropy_loss()
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_tensors.append(total_loss.detach())

        if (update + 1) % 300 == 0:
            elapsed = time.time() - started
            print(
                f"seed={seed} update={update + 1}/{EPOCHS} "
                f"loss={total_loss.item():.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    net.lstm.disable_sign_tracking()
    sampled_t, preferred_t = net.lstm.get_sign_history()
    loss = torch.stack(loss_tensors).cpu().numpy().astype(np.float32)
    sampled = sampled_t.cpu().numpy().astype(bool)
    preferred = preferred_t.cpu().numpy().astype(bool)
    assert sampled.shape == (EPOCHS, SEQ_LEN, HIDDEN_SIZE)
    assert preferred.shape == (EPOCHS, HIDDEN_SIZE)
    return net, loss, sampled, preferred


@torch.no_grad()
def evaluate(net, env):
    net.eval()
    performance = np.empty(len(env.envs), dtype=np.float32)
    for env_idx, task_env in enumerate(env.envs):
        correct = 0
        for _ in range(EVAL_TRIALS):
            task_env.new_trial()
            observation, ground_truth = task_env.ob, task_env.gt
            task_id = np.zeros((observation.shape[0], len(env.envs)))
            task_id[:, env_idx] = 1.0
            model_input = np.concatenate([observation, task_id], axis=1)
            model_input = torch.from_numpy(
                model_input[:, None, :]
            ).float().to(DEVICE)
            outputs, _, _ = net(model_input)
            prediction = outputs.argmax(dim=2)[-1, 0].item()
            correct += prediction == int(ground_truth[-1])
        performance[env_idx] = correct / EVAL_TRIALS
    return performance


def scalar_summary(seed_metrics, performance):
    scalar_keys = [
        key for key, value in seed_metrics[0].items()
        if np.isscalar(value)
    ]
    summary = {
        "protocol": {
            "task_collection": "yang19",
            "n_tasks": int(performance.shape[1]),
            "n_seeds": N_SEEDS,
            "updates": EPOCHS,
            "sequence_length": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "learning_rate": LEARNING_RATE,
            "gumbel_temperature": 0.1,
            "evaluation_trials_per_task": EVAL_TRIALS,
            "early_and_late_window_updates": int(EPOCHS * EARLY_FRACTION),
            "device": str(DEVICE),
        },
        "metrics": {},
        "mean_task_accuracy": {
            "per_seed": performance.mean(axis=1).tolist(),
            "mean": float(performance.mean(axis=1).mean()),
            "sd_across_seeds": float(performance.mean(axis=1).std(ddof=1)),
        },
    }
    for key in scalar_keys:
        values = np.asarray([metrics[key] for metrics in seed_metrics], dtype=float)
        summary["metrics"][key] = {
            "per_seed": values.tolist(),
            "mean": float(values.mean()),
            "sd_across_seeds": float(values.std(ddof=1)),
        }
    return summary


def main():
    print(f"device={DEVICE}", flush=True)
    losses = []
    packed_samples = []
    preferred_signs = []
    performances = []
    metrics_by_seed = []
    task_names = None

    for seed in range(N_SEEDS):
        env = make_environment()
        if task_names is None:
            task_names = [
                task_env.spec.id.split("yang19.")[1].split("-v0")[0]
                for task_env in env.envs
            ]
        net, loss, sampled, preferred = train_one_seed(seed, env)
        metrics = flip_metrics(sampled, preferred)
        performance = evaluate(net, env)

        losses.append(loss)
        packed_samples.append(np.packbits(sampled, axis=-1))
        preferred_signs.append(preferred)
        performances.append(performance)
        metrics_by_seed.append(metrics)
        print(
            f"seed={seed} accuracy={performance.mean():.4f} "
            f"sample_flip_early={metrics['sample_flip_rate_early_10pct']:.4f} "
            f"sample_flip_late={metrics['sample_flip_rate_late_10pct']:.4f} "
            f"preferred_flip_all={metrics['preferred_flip_rate_per_update_neuron']:.6f}",
            flush=True,
        )

    losses = np.stack(losses)
    packed_samples = np.stack(packed_samples)
    preferred_signs = np.stack(preferred_signs)
    performances = np.stack(performances)
    per_update_sample = np.stack(
        [m["per_update_sample_flip_rate"] for m in metrics_by_seed]
    )
    per_update_disagreement = np.stack(
        [m["per_update_disagreement_rate"] for m in metrics_by_seed]
    )
    per_update_preferred = np.stack(
        [m["per_update_preferred_flip_rate"] for m in metrics_by_seed]
    )
    last_preferred_flip = np.stack(
        [m["last_preferred_flip_update"] for m in metrics_by_seed]
    )
    preferred_flip_counts = np.stack(
        [m["preferred_flips_per_neuron"] for m in metrics_by_seed]
    )

    result_dir = Path("result/multitask")
    output_dir = Path("output/multitask")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        result_dir / "eisep_sign_flip_results.npz",
        loss=losses,
        sampled_signs_packbits=packed_samples,
        preferred_signs=preferred_signs,
        performance=performances,
        task_names=np.asarray(task_names),
        per_update_sample_flip_rate=per_update_sample,
        per_update_disagreement_rate=per_update_disagreement,
        per_update_preferred_flip_rate=per_update_preferred,
        last_preferred_flip_update=last_preferred_flip,
        preferred_flip_counts=preferred_flip_counts,
        hidden_size=HIDDEN_SIZE,
    )

    summary = scalar_summary(metrics_by_seed, performances)
    summary_path = result_dir / "eisep_sign_flip_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
