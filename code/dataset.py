import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def create_window(data, start_idx, window_size):
    end_idx = start_idx + window_size
    if end_idx <= len(data):
        return data[start_idx:end_idx]
    else:
        # Not enough data from start_idx to end; need to wrap and repeat
        remaining_length = window_size - (len(data) - start_idx)
        repeats = (remaining_length // len(data)) + 1
        extended_data = np.tile(data, (repeats, 1))
        return np.concatenate((data[start_idx:], extended_data[:remaining_length]), axis=0)


class IMUDataset(Dataset):
    def __init__(self, root_dir, window_size=128, features=6,
                 subjects_per_batch=32, files_per_subject=4, transform=None):
        """
        Args:
            root_dir (str): Path to the processed_data folder containing S001 to S400.
            window_size (int): Number of rows to extract from each CSV file.
            features (int): Number of features per row (should be 6).
            subjects_per_batch (int): Number of subject folders to sample per batch.
            files_per_subject (int): Number of CSV files to sample per subject.
            transform (callable, optional): Optional transform to apply to each sample.
        """
        self.root_dir = root_dir
        self.window_size = window_size
        self.features = features
        self.subjects_per_batch = subjects_per_batch
        self.files_per_subject = files_per_subject
        self.transform = transform

        # List all subject folders that match the pattern "S###"
        self.subject_dirs = sorted([
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('S')
        ])

        # Build a dictionary mapping subject folder to its CSV file paths (searching recursively)
        self.subject_files = {}
        for subj_dir in self.subject_dirs:
            csv_files = []
            for root, dirs, files in os.walk(subj_dir):
                for file in files:
                    if file.lower().endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            # Only keep subjects with at least the required number of CSV files
            if len(csv_files) >= self.files_per_subject:
                self.subject_files[subj_dir] = csv_files

        # Filter subject_dirs to only include those with sufficient CSV files
        self.subject_dirs = [s for s in self.subject_dirs if s in self.subject_files]

        # Define an arbitrary dataset length since we sample randomly each time.
        self._length = 10000

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # We ignore idx and sample a new batch each time.
        # Step 1: Randomly select subjects_per_batch subject folders.
        selected_subjects = np.random.choice(self.subject_dirs, self.subjects_per_batch, replace=False)
        samples = []

        for subj in selected_subjects:
            # Randomly choose files_per_subject CSV files for this subject.
            files = self.subject_files[subj]
            selected_files = np.random.choice(files, self.files_per_subject, replace=False)

            for csv_file in selected_files:
                # Load CSV data. We assume no header; adjust as needed.
                df = pd.read_csv(csv_file, header=None)
                data = df.apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
                n = data.shape[0]
                # Randomly choose a pivot index.
                i = np.random.randint(1, n)
                # Extract a continuous chunk of window_size rows with wrap-around if necessary.
                window = create_window(data,i,self.window_size)
                # Convert the window to a tensor (shape: window_size x features).
                sample = torch.from_numpy(window)
                if self.transform:
                    sample = self.transform(sample)
                samples.append(sample)

        # Stack all samples into a batch tensor with shape (subjects_per_batch * files_per_subject, window_size, features)
        batch = torch.stack(samples, dim=0)
        return batch

# Example usage:
if __name__ == '__main__':
    # Path to your processed_data folder
    root_dir = r'C:\CS-671_project\Data_Processed\Self-supervised_Training_data_arranged'
    dataset = IMUDataset(root_dir=root_dir, window_size=128, features=6,
                         subjects_per_batch=32, files_per_subject=4)

    # Create a DataLoader with num_workers = 4 and pin_memory enabled for efficient GPU transfer.
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # Iterate through one batch
    for batch in dataloader:
        # Each __getitem__ returns a batch of shape: (32 * 4, 128, 6) i.e., (128, 128, 6)
        batch = batch.squeeze(0)  # Remove the extra batch dimension from DataLoader
        print("Batch shape:", batch.shape)

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = batch.to(device)
        # Now you can use `batch` in your training pipeline.
        break
