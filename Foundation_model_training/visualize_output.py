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
    parser = argparse.ArgumentParser('Visualize MAE Reconstruction', add_help=False)
    parser.add_argument('--checkpoint', type=str, default=None, help='path to model checkpoint')
    parser.add_argument('--data_path', type=str, default=None, help='path to data for visualization')
    parser.add_argument('--output_dir', type=str, default='./visualizations', help='output directory for plots')
    parser.add_argument('--num_samples', type=int, default=5, help='number of samples to visualize')
    parser.add_argument('--channels', type=int, nargs='+', default=[0, 1, 2], help='channels to visualize')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='device to run inference on')
    parser.add_argument('--autoencoder_path', type=str, 
                       default='/home/teaching/Documents/G28_dl/Human-Gait-Representation-Learning/code_ds/autoencoder_results_2nd/autoencoder_final.pth',
                       help='path to the pretrained autoencoder model')
    return parser.parse_args()

def visualize_reconstructions(model, dataloader, args, config):
    """Generate and visualize reconstructions from the model."""
    model.eval()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get a batch of data
    all_samples = []
    for batch in dataloader:
        all_samples.append(batch)
        if len(all_samples) * batch.shape[0] >= args.num_samples:
            break

    # print("✅ Samples.shape: ",all_samples[0].shape)
    
    all_samples = torch.cat(all_samples, dim=0)[:args.num_samples]
    all_samples = all_samples.to(device)
    # print("✅ All samples.shape: ",all_samples.shape)
    
    # Run through model to get reconstructions
    with torch.no_grad():
        print("Generating reconstructions...")
        loss, pred, mask = model(all_samples, mask_ratio=config.mask_ratio)
        
        # Get the masked indices
        B = all_samples.shape[0]
        
        # Unpatchify the predictions to get the full reconstructions
        # Reshape predictions to match expected format [B, L, patch_size*channels]
        pred = pred.reshape(B, -1, config.patch_size * 128)
        # print("✅ Pred_reshaped: ",pred.shape)
        
        # Process original input through the same projection layers as used in the model
        input_projected = model.project(all_samples.permute(0, 2, 1))
        #print("✅ Input_projected: ",input_projected.shape)
        # Unpatchify predictions to get reconstructions in the same format as input_projected
        recon = model.unpatchify(pred)  # [B, C, T]
        #print("✅ Recon: ",recon.shape)

    # Visualization
    print("Creating visualizations...")
    channels_to_plot = [ch for ch in args.channels if ch < input_projected.shape[1]]
    
    for sample_idx in range(min(args.num_samples, all_samples.shape[0])):
        plt.figure(figsize=(12, 8))
        
        for i, channel in enumerate(channels_to_plot):
            plt.subplot(len(channels_to_plot), 1, i+1)
            
            # Get original and reconstruction for this channel
            original = input_projected[sample_idx, channel].cpu().numpy()
            reconstruction = recon[sample_idx, channel].cpu().numpy()
            
            # Plot
            plt.plot(original, label=f'Original (Channel {channel})', alpha=0.7)
            plt.plot(reconstruction, label=f'Reconstruction (Channel {channel})', alpha=0.7, linestyle='--')
            
            plt.legend()
            plt.title(f'Sample {sample_idx+1}, Channel {channel}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'sample_{sample_idx+1}_reconstruction.png'))
        plt.close()
    
    # Create a comparison grid for all samples and channels
    fig, axs = plt.subplots(args.num_samples, len(channels_to_plot), 
                            figsize=(4*len(channels_to_plot), 3*args.num_samples))
    
    if args.num_samples == 1 and len(channels_to_plot) == 1:
        axs = np.array([[axs]])
    elif args.num_samples == 1:
        axs = axs.reshape(1, -1)
    elif len(channels_to_plot) == 1:
        axs = axs.reshape(-1, 1)
    
    for sample_idx in range(min(args.num_samples, all_samples.shape[0])):
        for i, channel in enumerate(channels_to_plot):
            ax = axs[sample_idx, i]
            
            # Get original and reconstruction for this channel
            original = input_projected[sample_idx, channel].cpu().numpy()
            reconstruction = recon[sample_idx, channel].cpu().numpy()
            
            # Plot
            ax.plot(original, label='Original', alpha=0.7)
            ax.plot(reconstruction, label='Reconstruction', alpha=0.7, linestyle='--')
            
            # Calculate correlation
            corr = np.corrcoef(original, reconstruction)[0, 1]
            # print(f'{i},th correlation: {corr:.3f}')
            
            ax.set_title(f'Sample {sample_idx+1}, Ch {channel}, Corr: {corr:.2f}')
            ax.grid(True, alpha=0.3)
            
            if sample_idx == 0 and i == 0:
                ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'all_samples_grid.png'))
    plt.close()
    
    print(f"Visualizations saved to {args.output_dir}")

def main():
    args = parse_args()
    
    # Load config
    config = Config_MBM_EEG()
    
    # Override with command line args if provided
    if args.data_path is not None:
        config.val_path = args.data_path
    
    print(f"Loading data from: {config.val_path}")
    
    # Load model and data
    device = torch.device(args.device)
    
    # Check if autoencoder path exists
    autoencoder_path = args.autoencoder_path
    if not os.path.exists(autoencoder_path):
        print(f"Warning: Autoencoder path {autoencoder_path} not found.")
        print("Searching for autoencoder in alternative locations...")
        
        # Try relative path
        alt_path = os.path.join(os.path.dirname(__file__), 'autoencoder_results/autoencoder_final.pth')
        if os.path.exists(alt_path):
            autoencoder_path = alt_path
            print(f"Found autoencoder at: {autoencoder_path}")
        else:
            # Try looking for autoencoder_best.pth instead
            alt_path = os.path.join(os.path.dirname(__file__), 'autoencoder_results/autoencoder_best.pth')
            if os.path.exists(alt_path):
                autoencoder_path = alt_path
                print(f"Found alternative autoencoder model at: {autoencoder_path}")
            else:
                print("No autoencoder model found. Continuing without autoencoder.")
                autoencoder_path = None
    
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
        mlp_ratio=config.mlp_ratio
        # autoencoder_path=autoencoder_path
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
    dataloader = get_dataloader(root_dir=config.root_path, batch_size=args.num_samples)
    
    # Run visualization
    visualize_reconstructions(model, dataloader, args, config)

if __name__ == "__main__":
    main()