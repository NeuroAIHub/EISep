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
print(f"Using device: {DEVICE}", flush=True)

class SortDataset(Dataset):
    
    def __init__(self, seq_len=5, num_samples=10000):
        self.seq_len = seq_len
        self.num_samples = num_samples
        numbers = torch.rand(num_samples, seq_len)
        argsort = torch.argsort(numbers, dim=1)
        
        self.data = numbers.unsqueeze(-1) # Shape: (num_samples, seq_len, 1)
        self.target = argsort # Shape: (num_samples, seq_len)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def train_and_evaluate(config):
    
    model_name, seed, lr, epochs, batch_size, hidden_size, seq_len, wd = config

    print(f"Starting: model={model_name}, seed={seed}, lr={lr:.5f}, wd={wd}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = SortDataset(seq_len=seq_len, num_samples=50000)
    test_dataset = SortDataset(seq_len=seq_len, num_samples=1000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = 1 
    output_size = seq_len**2 
    model = init_model(input_size, hidden_size, output_size, model_name).to(DEVICE)
    
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    loader = iter(train_loader)
    for i in range(epochs):
        try:
            inputs, targets = next(loader)
        except StopIteration:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            loader = iter(train_loader)
            inputs, targets = next(loader)
        
        inputs = inputs.to(DEVICE).transpose(0, 1) 
        targets = targets.to(DEVICE) 

        optimizer.zero_grad()

        outputs, eimask_logit, _ = model(inputs)
        
        B, T = targets.shape
        
        predictions = outputs[-1].reshape(B, T, T).flatten(0,1)
        targets_flat = targets.flatten(0,1)
        decoding_loss = criterion(predictions, targets_flat)

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
    total_test_acc = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            batch_inputs = batch_inputs.to(DEVICE).transpose(0, 1)
            batch_targets = batch_targets.to(DEVICE)
            
            outputs, _, _ = model(batch_inputs) # (seq, batch, vocab_size)
            
            B, T = batch_targets.shape
            predections = outputs[-1].reshape(B, T, T).flatten(0,1)
            target = batch_targets.flatten(0,1)
            predict_label = torch.argmax(predections, dim=1)
            accuracy = (predict_label == target).float().mean().item()
            
            total_test_acc += accuracy

    avg_test_acc = total_test_acc / len(test_loader)
    print(f"Finished: model={model_name}, seed={seed}, lr={lr:.5f}, Test Accuracy: {avg_test_acc:.4f}", flush=True)

    model_dir = './models/sort_problem'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}_seed{seed}_lr{lr:.5f}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}", flush=True)

    return {
        'model_name': model_name,
        'seed': seed,
        'lr': lr,
        'test_accuracy': avg_test_acc
    }

def main():

    MODEL_NAMES = ['eisep', 'simple', 'dann_sgd', 'dann_adam', 'colei_sgd', 'colei_adam', 'dbpn']
    SEEDS = [0, 42, 3407, 998244353, 1000000007]
    LEARNING_RATES = [2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.2e-2, 2.5e-2, 5e-2, 8e-2]
    WEIGHT_DECAY = 1e-5
    EPOCHS = 3000
    BATCH_SIZE = 64
    HIDDEN_SIZE = 128
    SEQ_LEN = 5 
    NUM_PROCESSES = 10

    configs = list(product(MODEL_NAMES, SEEDS, LEARNING_RATES, [EPOCHS], [BATCH_SIZE], [HIDDEN_SIZE], [SEQ_LEN], [WEIGHT_DECAY]))
    print(f"Total number of experiments to run: {len(configs)}", flush=True)
    print(f"Using {NUM_PROCESSES} parallel processes.", flush=True)


    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(train_and_evaluate, configs)

    results_dir = './results/sort_problem'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'benchmark_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nAll experiments completed. Results saved to {results_path}", flush=True)

    print("\n--- Best Performing Learning Rate per Model (based on Accuracy) ---", flush=True)
    for model_name in MODEL_NAMES:
        model_results = [r for r in results if r['model_name'] == model_name]
        if not model_results:
            continue

        lr_performance = {}
        for lr in LEARNING_RATES:
            lr_results = [r['test_accuracy'] for r in model_results if abs(r['lr'] - lr) < 1e-9]
            if lr_results:
                lr_performance[lr] = np.mean(lr_results)

        if not lr_performance:
            continue
            
        best_lr = max(lr_performance, key=lr_performance.get)
        best_acc = lr_performance[best_lr]
        print(f"Model: {model_name:12s} | Best LR: {best_lr:.5f} | Avg. Accuracy: {best_acc:.4f}", flush=True)


if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
