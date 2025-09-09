import os
import sys
import time
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
# Import configuration and modules from your project files.
from Foundation_model_training.config import Config_MBM_EEG
from Foundation_model_training.dataset import get_dataloader  # Updated: use IMUDataset from dataset.py
from Foundation_model_training.mae import MAEforEEG
from Foundation_model_training.trainer import train_one_epoch, evaluate, NativeScalerWithGradNormCount as NativeScaler
from Foundation_model_training.utils import adjust_learning_rate, save_model

# =============================================================================
# Training Script for EEG/IMU Pretraining with MAE
# =============================================================================

def get_args_parser():
    parser = argparse.ArgumentParser('EEG MAE Pretraining', add_help=False)
    
    # Training parameters
    parser.add_argument('--num_epoch', default=300, type=int, help='number of training epochs')
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
    parser.add_argument('--projection_type', default='autoencoder', type=str, 
                        choices=['autoencoder', 'conv', 'linear', 'attention'],
                        help='method to project from 6 to 128 channels')
    
    # Data parameters
    parser.add_argument('--data_path', default='path/to/processed_data', type=str, help='path to processed_data folder')
    # For our dataset, we assume the IMUDataset class uses its own strategy to load CSV files.
    
    # Output and checkpointing
    parser.add_argument('--output_path', default='./results/imu_pretrain', type=str, help='output folder for results and checkpoints')
    
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

    # Parse command line arguments to override config
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Set projection type from args if provided
    projection_type = args.projection_type if hasattr(args, 'projection_type') else 'conv'
    
    # Create output directory with a timestamp for saving checkpoints.
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    output_dir = os.path.join(config.output_path, f'imu_pretrain_{projection_type}', timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save config details for reproducibility.
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        for attr, value in config.__dict__.items():
            f.write(f"{attr}: {value}\n")
        f.write(f"projection_type: {projection_type}\n")

    # Create DataLoader
    dataloader = get_dataloader(root_dir=config.root_path)

    # create Validation dataset and dataloader
    val_loader = get_dataloader(root_dir=config.val_path)

    # Update model creation to use the projection type
    model=MAEforEEG(
            time_len=212,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            decoder_embed_dim=config.decoder_embed_dim,
            decoder_depth=8,
            decoder_num_heads=config.decoder_num_heads,
            mlp_ratio=config.mlp_ratio,
        )

    

    # Load pretrained weights
    pretrained_path = '/home/teaching/Documents/G28_dl/cs671-p07/pretrained_weight/mae_checkpoint.pth'
    if os.path.exists(pretrained_path):
        print(f"\n📦 Loading pre-trained weights from {pretrained_path}...")
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)

        if 'model' in checkpoint:
            checkpoint = checkpoint['model']  # handle nested state_dict

        model_state = model.state_dict()
        
        # Create filtered checkpoint with matching shapes
        filtered_checkpoint = {}
        for k, v in checkpoint.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_checkpoint[k] = v
                else:
                    print(f"⏭️ Skipping {k} due to size mismatch: {v.shape} vs {model_state[k].shape}")
            else:
                print(f"⏭️ Skipping {k} (not found in model)")

        # Load filtered weights
        missing_keys, unexpected_keys = model.load_state_dict(filtered_checkpoint, strict=False)
        
        print("\n✅ Pretrained weights loaded.")
        loaded_keys = model_state.keys() & filtered_checkpoint.keys()
        print(f"\n🔹 Total weights loaded: {len(loaded_keys)}")
    else:
        print(f"⚠️ Pretrained weight file not found at {pretrained_path}. Skipping weight loading.")

    model.to(device)
    # print("-------------MODEL------------")
    # print(model)
    model_without_ddp = model

    # Optimizer and loss scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    # Training loop
    print("Starting IMU MAE pretraining...")
    start_time = time.time()
    # cor_list = []

    # enable interactive mode
    plt.ion()
    fig, ax = plt.subplots()
    train_losses = []
    val_losses   = []
    val_epochs   = []

    for epoch in range(config.num_epoch):
        current_lr = adjust_learning_rate(optimizer, epoch, config)
        print(f"Epoch {epoch+1}/{config.num_epoch} | LR: {current_lr:.6f}")

        # Train for one epoch using adapted data
        train_loss = train_one_epoch(model, dataloader, optimizer, device, epoch,
                              loss_scaler, log_writer=None, config=config, start_time=start_time,
                              model_without_ddp=model_without_ddp)
        train_losses.append(train_loss)

        # --- validation every 5 epochs ---
        if (epoch) % 5 == 0:
            val_loss = evaluate(model, val_loader, device, config)
            val_losses.append(val_loss)
            val_epochs.append(epoch + 1)

                        # --- plot and save every 10 epochs ---
        if (epoch) % 5 == 0 or (epoch + 1 == config.num_epoch):
            ax.clear()
            ax.plot(range(1, epoch + 2), train_losses, label='Train Loss')
            if val_epochs:
                ax.plot(val_epochs, val_losses, 'o-', label='Val Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (log scale)')
            ax.set_yscale('log')

            ax.set_title('Training vs Validation Loss')
            ax.legend()

            plot_path = os.path.join(output_dir, f'loss_curve_epoch_{epoch+1}.png')
            plt.savefig(plot_path, dpi=300)
            print(f"📈 Loss curve saved to {plot_path}")


        # cor_list.append(cor)
        # print(f"Epoch {epoch+1}: Average correlation: {cor:.4f}")

        # Save checkpoint
        if (epoch+1 % 50 == 0) or (epoch + 1 == config.num_epoch):
            checkpoint_dir = os.path.join(output_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_model(config, epoch, model_without_ddp, optimizer, loss_scaler, checkpoint_dir)
            print(f"Checkpoint saved at epoch {epoch+1}.")

        # Freeze the project layer after 80 epochs
        if epoch == 80:
            for param in model.project.parameters():
                param.requires_grad = False
            print("\n🧊 Project layer frozen after 80 epochs.")

    plt.ioff()  # turn interactive mode off

    # Final save
    total_time = time.time() - start_time
    print(f"Training completed in {str(datetime.timedelta(seconds=int(total_time)))}")
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    torch.save(model_without_ddp.state_dict(), final_model_path)
    print(f"Final model weights saved at {final_model_path}")
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))


if __name__ == '__main__':
    # Create the configuration object from config.py
    config = Config_MBM_EEG()
    # Override config parameters from command-line arguments, if provided.
    parser = get_args_parser()
    args = parser.parse_args()
    # For now, we use the config as defined.
    main(config)

