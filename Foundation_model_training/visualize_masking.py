import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm

# Import local modules
from Foundation_model_training.config import Config_MBM_EEG
from Foundation_model_training.dataset import get_dataloader, IMUDataset
from Foundation_model_training.mae import MAEforEEG
from Foundation_model_training.utils import load_model

def parse_args():
    parser = argparse.ArgumentParser('Visualize MAE Masking', add_help=False)
    parser.add_argument('--checkpoint', type=str, default=None, help='path to model checkpoint')
    parser.add_argument('--data_path', type=str, default=None, help='path to data for visualization')
    parser.add_argument('--output_dir', type=str, default='./masking_visualizations', help='output directory for plots')
    parser.add_argument('--num_samples', type=int, default=3, help='number of samples to visualize')
    parser.add_argument('--channels', type=int, nargs='+', default=[0, 1, 2], help='channels to visualize')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='ratio of patches to mask')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='device to run inference on')
    return parser.parse_args()

def get_masked_output(model, x, mask_ratio):
    """Apply masking similar to how the model does it internally."""
    # First apply projection to input (similar to what model does)
    x_proj = model.project(x.permute(0, 2, 1))
    
    # Run the encoder part with provided mask_ratio
    latent, mask, ids_restore = model.forward_encoder(x.permute(0, 2, 1), mask_ratio)
    
    # Apply the mask manually to the projected input
    B, C, T = x_proj.shape
    p = model.patch_embed.patch_size
    
    # Convert to patches
    x_patch = x_proj.reshape(B, C, T // p, p).permute(0, 2, 1, 3).reshape(B, T // p, C * p)
    
    # Create masked version by zeroing out masked positions
    masked_patches = x_patch.clone()
    
    # Create inverse mask (0 for kept, 1 for masked)
    inv_mask = torch.ones_like(mask) - mask
    
    # Convert mask to right shape
    mask_reshape = inv_mask.unsqueeze(-1).repeat(1, 1, x_patch.shape[-1])
    
    # Apply mask
    masked_patches = masked_patches * mask_reshape
    
    # Convert back to time-series format
    masked_series = masked_patches.reshape(B, T // p, C, p).permute(0, 2, 1, 3).reshape(B, C, T)
    
    return x_proj, masked_series, mask

def visualize_masking(model, dataloader, args, config):
    """Visualize the masking and reconstruction process."""
    model.eval()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get a batch of data
    all_samples = []
    for batch in dataloader:
        all_samples.append(batch)
        if len(all_samples) * batch.shape[0] >= args.num_samples:
            break
    
    all_samples = torch.cat(all_samples, dim=0)[:args.num_samples]
    all_samples = all_samples.to(device)
    
    # Run inference with visualization of masking
    with torch.no_grad():
        # Get the original, masked, and reconstructed versions
        original_proj, masked_series, mask = get_masked_output(model, all_samples, args.mask_ratio)
        
        # Get the reconstruction
        loss, pred, _ = model(all_samples, mask_ratio=args.mask_ratio)
        
        # Reshape and unpatchify the predictions
        B = all_samples.shape[0]
        pred = pred.reshape(B, -1, config.patch_size * model.project[0].out_channels)
        recon = model.unpatchify(pred)
    
    # Visualization
    print("Creating masking visualizations...")
    channels_to_plot = [ch for ch in args.channels if ch < original_proj.shape[1]]
    
    for sample_idx in range(min(args.num_samples, all_samples.shape[0])):
        plt.figure(figsize=(15, 10))
        
        # Calculate the percentage of masked values
        mask_percentage = (1.0 - mask[sample_idx].float().mean().item()) * 100
        
        plt.suptitle(f"Sample {sample_idx+1} - {mask_percentage:.1f}% masked", fontsize=16)
        
        for i, channel in enumerate(channels_to_plot):
            plt.subplot(len(channels_to_plot), 1, i+1)
            
            # Get data for this channel
            original = original_proj[sample_idx, channel].cpu().numpy()
            masked = masked_series[sample_idx, channel].cpu().numpy()
            reconstruction = recon[sample_idx, channel].cpu().numpy()
            
            # Calculate correlation between original and reconstruction
            corr = np.corrcoef(original, reconstruction)[0, 1]
            
            # Plot
            time_axis = np.arange(len(original))
            plt.plot(time_axis, original, label='Original', alpha=0.7)
            plt.plot(time_axis, masked, label='Masked Input', alpha=0.5, linestyle=':')
            plt.plot(time_axis, reconstruction, label='Reconstruction', alpha=0.7, linestyle='--')
            
            plt.title(f'Channel {channel}, Correlation: {corr:.4f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(args.output_dir, f'sample_{sample_idx+1}_masking.png'))
        plt.close()
    
    # Create a visualization of the mask itself
    for sample_idx in range(min(args.num_samples, all_samples.shape[0])):
        # Get mask for this sample
        sample_mask = mask[sample_idx].cpu().numpy()
        
        plt.figure(figsize=(10, 3))
        plt.imshow(sample_mask.reshape(1, -1), cmap='binary', aspect='auto')
        plt.title(f'Sample {sample_idx+1} Mask Pattern (white = kept, black = masked)')
        plt.colorbar(label='Keep (1) / Mask (0)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'sample_{sample_idx+1}_mask_pattern.png'))
        plt.close()
    
    print(f"Visualizations saved to {args.output_dir}")

def main():
    args = parse_args()
    
    # Load config
    config = Config_MBM_EEG()
    
    # Override with command line args if provided
    if args.data_path is not None:
        config.val_path = args.data_path
    
    # Also override mask_ratio if provided
    config.mask_ratio = args.mask_ratio
    
    print(f"Loading data from: {config.val_path}")
    print(f"Using mask ratio: {config.mask_ratio}")
    
    # Load model and data
    device = torch.device(args.device)
    
    # Create model
    model = MAEforEEG(
        time_len=212,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=8,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio,
        img_recon_weight=config.img_recon_weight,
        use_nature_img_loss=config.use_nature_img_loss
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        print("No checkpoint provided. Using random weights.")
    
    model = model.to(device)
    
    # Create dataloader
    dataloader = get_dataloader(root_dir=config.val_path, batch_size=args.num_samples)
    
    # Run visualization
    visualize_masking(model, dataloader, args, config)

if __name__ == "__main__":
    main() 