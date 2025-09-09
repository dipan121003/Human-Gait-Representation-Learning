import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
from pathlib import Path

def create_window(data, start_idx, window_size):
    end_idx = start_idx + window_size
    if end_idx <= len(data):
        return data[start_idx:end_idx]
    else:
        remaining = window_size - (len(data) - start_idx)
        return np.concatenate((data[start_idx:], data[:remaining]), axis=0)

class IMUDataset(Dataset):
    def __init__(self, root_dir, window_size=212, seed=None):
        self.window_size = window_size
        self.seed = seed
        self.rng = np.random.default_rng(seed) if seed else None
        
        # Collect and sort CSV paths for reproducibility
        self.csv_paths = []
        # Collect all CSV paths recursively
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith('.csv'):
                    self.csv_paths.append(os.path.join(root, file))
        if not self.csv_paths:
            raise ValueError(f"No CSV files found in {root_dir}")

        # Precompute file metadata (path, length) 
        self.file_info = []
        for path in self.csv_paths:
            with open(path) as f:
                n_rows = sum(1 for _ in f) - 1  # Exclude header if present
            self.file_info.append((path, max(n_rows, window_size)))  # Ensure min length

        # Calculate cumulative indices for deterministic access
        self.cumulative_windows = np.cumsum([
            (length - window_size + 1) if length >= window_size else 1 
            for _, length in self.file_info
        ])
        self.total_windows = self.cumulative_windows[-1] if self.cumulative_windows.size > 0 else 0

    def __len__(self):
        return len(self.csv_paths)

    def _load_data(self, path):
        """Cached data loading with proper NaN handling"""
        df = pd.read_csv(path, header=None)
        data = df.apply(pd.to_numeric, errors='coerce').values.astype(np.float32)

        # Replace NaN handling with z-score normalization
        # Handle division by zero in normalization
        data = (data - np.nanmean(data, axis=0)) / np.nanstd(data, axis=0)


        data = np.nan_to_num(data, nan=0.0)
        return data

    def __getitem__(self, idx):
        # Use seed if provided for reproducibility
        if self.seed:
            local_rng = np.random.default_rng(self.seed + idx)
        else:
            local_rng = np.random.default_rng()
            
        # Map index to file and position
        idx = idx % self.total_windows  # Cycle through available windows
        file_idx = np.searchsorted(self.cumulative_windows, idx, side='right')
        
        if file_idx > 0:
            idx -= self.cumulative_windows[file_idx-1]
            
        path, total_length = self.file_info[file_idx]
        data = self._load_data(path)
        
        # Calculate valid start index with wrapping
        if total_length >= self.window_size:
            max_start = total_length - self.window_size
            start_idx = idx % max_start if max_start > 0 else 0
        else:
            start_idx = local_rng.integers(0, len(data))  # For short files
            
        window = create_window(data, start_idx, self.window_size)
        return torch.tensor(window, dtype=torch.float32)

def get_dataloader(root_dir, batch_size=128, seed=None, num_workers=4):
    # Seed configuration for reproducibility
    def init_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataset = IMUDataset(root_dir, window_size=212, seed=seed)
    
    generator = torch.Generator()
    if seed:
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=init_worker,
        generator=generator,
        persistent_workers=True
    )




if __name__ == '__main__':
    # Usage
    root_dir = r"/home/teaching/Documents/G28_dl/cs671-p07/IMU_Dataset/Dataset/Self_supervise_data2/train"
    dataloader = get_dataloader(root_dir)

    # Verify structure
    sample_batch = next(iter(dataloader))
    print(f"Batch shape: {sample_batch.shape}")  # Should be [128, 6, 128]

