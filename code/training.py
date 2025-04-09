import os
import sys
import time
import datetime
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

# Import configuration and modules from your project files.
from config import Config_MBM_EEG
from dataset import IMUDataset  # Updated: use IMUDataset from dataset.py
from mae import MAEforEEG, PatchEmbed1D
from trainer import train_one_epoch, NativeScalerWithGradNormCount as NativeScaler
from utils import adjust_learning_rate, save_model

class IMUAdapter(nn.Module):
    def __init__(self, out_chans=128, out_time=512):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv1d(6, out_chans, kernel_size=3, padding=1),  # Converts 6 channels to 128 channels
            nn.ReLU(),
            nn.Conv1d(out_chans, out_chans, kernel_size=3, padding=1)
        )
        self.upsample = nn.Upsample(size=out_time, mode='nearest', align_corners=False)  # Upsample time dimension from 128 to out_time (512)

    def forward(self, x):
        # x: [B, 128, 6] where 128 is timestamp dimension and 6 is channels.
        # Permute to [B, 6, 128] for Conv1d.
        x = x.permute(0, 2, 1)
        x = self.project(x)       # Now x is [B, 128, 128]
        x = self.upsample(x)      # Now x is [B, 128, 512]
        x = x.permute(0, 2, 1)    # Now x is [B, 512, 128]
        return x

# =============================================================================
# Training Script for EEG/IMU Pretraining with MAE
# =============================================================================

def get_args_parser():
    parser = argparse.ArgumentParser('EEG MAE Pretraining', add_help=False)
    
    # Training parameters
    parser.add_argument('--num_epoch', default=10, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size (number of samples per batch)')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='minimum learning rate after decay')
    parser.add_argument('--clip_grad', default=1.0, type=float, help='gradient clipping value')

    
    # Model parameters
    parser.add_argument('--time_len', default=128, type=int, help='length of the time series')
    parser.add_argument('--patch_size', default=4, type=int, help='patch size along time dimension')
    parser.add_argument('--in_chans', default=6, type=int, help='number of input channels (features)')
    parser.add_argument('--embed_dim', default=1024, type=int, help='dimension of patch embedding')
    parser.add_argument('--decoder_embed_dim', default=512, type=int, help='decoder embedding dimension')
    parser.add_argument('--depth', default=12, type=int, help='number of encoder transformer blocks')
    parser.add_argument('--num_heads', default=8, type=int, help='number of attention heads in encoder')
    parser.add_argument('--decoder_depth', default=4, type=int, help='number of decoder transformer blocks')
    parser.add_argument('--decoder_num_heads', default=8, type=int, help='number of attention heads in decoder')
    parser.add_argument('--mlp_ratio', default=4.0, type=float, help='MLP ratio in transformer blocks')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='ratio of patches to mask')
    
    # Data parameters
    parser.add_argument('--data_path', default='path/to/processed_data', type=str, help='path to processed_data folder')
    # For our dataset, we assume the IMUDataset class uses its own strategy to load CSV files.
    
    # Output and checkpointing
    parser.add_argument('--output_path', default='./results/eeg_pretrain', type=str, help='output folder for results and checkpoints')
    
    # Distributed training parameters (if using multiple GPUs)
    parser.add_argument('--local_rank', default=0, type=int, help='local rank for distributed training')
    
    return parser

def main(config):
    # Set device (use single GPU for simplicity; extend for distributed training as needed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Update configuration for IMU dimensions
    config.time_len = 128     # Our IMU sample length
    config.in_chans = 6       # Number of features per time step

    # Create output directory with a timestamp for saving checkpoints.
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    output_dir = os.path.join(config.output_path, 'imu_pretrain', timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save config details for reproducibility.
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        for attr, value in config.__dict__.items():
            f.write(f"{attr}: {value}\n")

    # Create the dataset instance
    dataset = IMUDataset(
        root_dir=config.root_path,
        window_size=config.time_len,
        features=config.in_chans,
        subjects_per_batch=32,
        files_per_subject=4,
        transform=None
    )
    print(f"Dataset size: {len(dataset)}; Data length per sample: {config.time_len}")

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    # Create IMUAdapter and place it on device
    imu_adapter = IMUAdapter(out_chans=128, out_time=512).to(device)

    # Create the MAE model (expects [batch, 512, 128])
    model = MAEforEEG(
        time_len=512,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        in_chans=128,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=8,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio,
        img_recon_weight=config.img_recon_weight,
        use_nature_img_loss=config.use_nature_img_loss
    )
    model.to(device)
    model_without_ddp = model

    # Optimizer and loss scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    # Training loop
    print("Starting IMU MAE pretraining...")
    start_time = time.time()
    cor_list = []

    for epoch in range(config.num_epoch):
        current_lr = adjust_learning_rate(optimizer, epoch, config)
        print(f"Epoch {epoch+1}/{config.num_epoch} | LR: {current_lr:.6f}")

        # Wrap dataloader to insert IMUAdapter processing before training
        def adapted_dataloader():
            for batch in dataloader:
                batch = batch.to(device)  # shape: [B, 128, 6]
                batch = imu_adapter(batch)  # shape: [B, 512, 128]
                yield batch

        # Train for one epoch using adapted data
        cor = train_one_epoch(model, adapted_dataloader(), optimizer, device, epoch,
                              loss_scaler, log_writer=None, config=config, start_time=start_time,
                              model_without_ddp=model_without_ddp)
        cor_list.append(cor)
        print(f"Epoch {epoch+1}: Average correlation: {cor:.4f}")

        # Save checkpoint
        if (epoch % 20 == 0) or (epoch + 1 == config.num_epoch):
            checkpoint_dir = os.path.join(output_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_model(config, epoch, model_without_ddp, optimizer, loss_scaler, checkpoint_dir)
            print(f"Checkpoint saved at epoch {epoch+1}.")

    # Final save
    total_time = time.time() - start_time
    print(f"Training completed in {str(datetime.timedelta(seconds=int(total_time)))}")
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    torch.save(model_without_ddp.state_dict(), final_model_path)
    print(f"Final model weights saved at {final_model_path}")


if __name__ == '__main__':
    # Create the configuration object from config.py
    config = Config_MBM_EEG()
    # Override config parameters from command-line arguments, if provided.
    parser = get_args_parser()
    args = parser.parse_args()
    # For now, we use the config as defined.
    main(config)

