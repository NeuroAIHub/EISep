import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
import multiprocessing
import os
import pickle
from itertools import product

from net.colei import ColEI
from net.dann import DANN
from net.simple import SimpleNet
from net.eisep import EISepNet
from net.dbp import DBPN
from net.common import init_model

warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

class AddingProblemDataset(Dataset):
    """Dataset for the adding problem task."""
    def __init__(self, seq_len=20, num_samples=10000):
        self.seq_len = seq_len
        self.num_samples = num_samples

        numbers = np.random.uniform(0, 1, (num_samples, seq_len))
        indicators = np.zeros((num_samples, seq_len))
        for i in range(num_samples):
            marked_positions = np.random.choice(seq_len, size=2, replace=False)
            indicators[i, marked_positions] = 1

        self.data = torch.FloatTensor(np.stack([numbers, indicators], axis=2))
        self.targets = torch.FloatTensor((numbers * indicators).sum(axis=1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def train_and_evaluate(config):
    
    model_name, seed, lr, epochs, batch_size, hidden_size, seq_len, wd = config

    print(f"Starting: model={model_name}, seed={seed}, lr={lr:.5f}, wd={wd}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = AddingProblemDataset(seq_len=seq_len, num_samples=50000)
    test_dataset = AddingProblemDataset(seq_len=seq_len, num_samples=1000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def dataset_generator(loader):
        while True:
            for data in loader:
                yield data[0].numpy(), data[1].numpy()

    train_gen = dataset_generator(train_loader)

    input_size = 2
    output_size = 1
    model = init_model(input_size, hidden_size, output_size, model_name).to(DEVICE)
    model.train()

    if not isinstance(model, (DANN, ColEI)):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()

    for i in range(epochs):
        inputs, targets = next(train_gen)
        
        inputs = torch.from_numpy(inputs).float().to(DEVICE).transpose(0, 1)
        targets = torch.from_numpy(targets).float().to(DEVICE)

        if not isinstance(model, (DANN, ColEI)):
            optimizer.zero_grad()

        outputs, eimask_logit, _ = model(inputs)
        predictions = outputs[-1, :, 0]
        decoding_loss = criterion(predictions, targets)

        entropy_loss = 0.0
        if hasattr(model, 'get_ei_entropy_loss') and model.entropy_loss:
             entropy_loss = model.get_ei_entropy_loss()

        l1_loss = 0.0
        if isinstance(model, SimpleNet):
            l1_loss = model.lstm.Wrec_free.abs().mean()

        loss = decoding_loss + 1 * entropy_loss + l1_loss * 0

        loss.backward()
        if isinstance(model, DANN):
            model.update(lr=lr, max_grad_norm=10)
        elif isinstance(model, ColEI):
            model.update(lr=lr, max_grad_norm=10)
        elif isinstance(model, DBPN):
            model.update(optimizer)
        else:
            optimizer.step()

    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            batch_inputs = batch_inputs.to(DEVICE).transpose(0, 1)
            batch_targets = batch_targets.to(DEVICE)
            outputs, _, _ = model(batch_inputs)
            predictions = outputs[-1, :, 0]
            loss = criterion(predictions, batch_targets)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Finished: model={model_name}, seed={seed}, lr={lr:.5f}, Test MSE: {avg_test_loss:.6f}", flush=True)

    model_dir = './models/adding_problem'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}_seed{seed}_lr{lr:.5f}.pth")
    torch.save(model.state_dict(), model_path)
    # print(f"Model saved to {model_path}")

    return {
        'model_name': model_name,
        'seed': seed,
        'lr': lr,
        'test_mse': avg_test_loss
    }

def main():
    
    MODEL_NAMES = ['eisep', 'simple', 'dann_sgd', 'dann_adam', 'colei_sgd', 'colei_adam', 'dbpn']
    SEEDS = [0, 42, 3407, 998244353, 1000000007]
    LEARNING_RATES = [3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.2e-2, 2.5e-2, 5e-2, 8e-2]
    WEIGHT_DECAY = 1e-5 
    EPOCHS = 3000
    BATCH_SIZE = 64
    HIDDEN_SIZE = 64
    SEQ_LEN = 20
    NUM_PROCESSES = 10

    configs = list(product(MODEL_NAMES, SEEDS, LEARNING_RATES, [EPOCHS], [BATCH_SIZE], [HIDDEN_SIZE], [SEQ_LEN], [WEIGHT_DECAY]))
    print(f"Total number of experiments to run: {len(configs)}")
    print(f"Using {NUM_PROCESSES} parallel processes.")

    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(train_and_evaluate, configs)

    results_dir = './results/adding_problem'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'benchmark_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nAll experiments completed. Results saved to {results_path}")

    print("\n--- Best Performing Learning Rate per Model ---")
    for model_name in MODEL_NAMES:
        model_results = [r for r in results if r['model_name'] == model_name]
        if not model_results:
            continue

        lr_performance = {}
        for lr in LEARNING_RATES:
            lr_results = [r['test_mse'] for r in model_results if abs(r['lr'] - lr) < 1e-9]
            if lr_results:
                lr_performance[lr] = np.mean(lr_results)

        if not lr_performance:
            continue
            
        best_lr = min(lr_performance, key=lr_performance.get)
        best_mse = lr_performance[best_lr]
        print(f"Model: {model_name:12s} | Best LR: {best_lr:.5f} | Avg. MSE: {best_mse:.6f}")


if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
