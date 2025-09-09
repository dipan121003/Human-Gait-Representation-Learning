import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

# Import local modules
from Foundation_model_training.config import Config_MBM_EEG
from Foundation_model_training.dataset import get_dataloader
from Foundation_model_training.mae import MAEforEEG
from Foundation_model_training.utils import load_model

def parse_args():
    parser = argparse.ArgumentParser('Evaluate MAE Model', add_help=False)
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--data_path', type=str, default=None, help='path to test data')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='output directory for results')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='device to run inference on')
    parser.add_argument('--num_samples', type=int, default=None, help='limit number of samples (None = use all)')
    return parser.parse_args()

def evaluate_model(model, dataloader, args, config):
    """Evaluate model performance on test dataset."""
    model.eval()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Metrics to track
    all_losses = []
    all_mse = []
    all_correlations = []
    channel_correlations = [[] for _ in range(128)]  # Assuming 128 channels
    
    processed_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch = batch.to(device)
            
            # Forward pass
            loss, pred, mask = model(batch, mask_ratio=config.mask_ratio)
            
            # Get batch size
            B = batch.shape[0]
            
            # Reshape predictions and get original through projection
            pred = pred.reshape(B, -1, config.patch_size * 128)
            input_projected = model.project(batch.permute(0, 2, 1))
            
            # Unpatchify predictions to get reconstructions
            recon = model.unpatchify(pred)  # [B, C, T]
            
            # Calculate metrics
            all_losses.append(loss.item())
            
            # Calculate MSE per sample
            for i in range(B):
                original = input_projected[i].cpu().numpy()
                reconstruction = recon[i].cpu().numpy()
                
                # Overall MSE for this sample
                mse = mean_squared_error(original.flatten(), reconstruction.flatten())
                all_mse.append(mse)
                
                # Calculate correlation for each channel
                for ch in range(min(128, original.shape[0])):
                    if ch < original.shape[0]:
                        corr = np.corrcoef(original[ch], reconstruction[ch])[0, 1]
                        channel_correlations[ch].append(corr)
                
                # Overall correlation for this sample (average across channels)
                sample_corr = np.mean([np.corrcoef(original[ch], reconstruction[ch])[0, 1] 
                                     for ch in range(min(128, original.shape[0]))])
                all_correlations.append(sample_corr)
            
            processed_samples += B
            
            # Check if we've processed enough samples
            if args.num_samples is not None and processed_samples >= args.num_samples:
                break
    
    # Compute average metrics
    avg_loss = np.mean(all_losses)
    avg_mse = np.mean(all_mse)
    avg_correlation = np.mean(all_correlations)
    
    # Compute per-channel statistics
    channel_avg_corr = [np.mean(corrs) if corrs else 0 for corrs in channel_correlations]
    top_channels = np.argsort(channel_avg_corr)[-10:][::-1]  # Top 10 channels by correlation
    bottom_channels = np.argsort(channel_avg_corr)[:10]  # Bottom 10 channels by correlation
    
    # Save results
    results = {
        'avg_loss': avg_loss,
        'avg_mse': avg_mse,
        'avg_correlation': avg_correlation,
        'channel_avg_corr': channel_avg_corr,
        'top_channels': top_channels.tolist(),
        'top_channels_corr': [channel_avg_corr[ch] for ch in top_channels],
        'bottom_channels': bottom_channels.tolist(),
        'bottom_channels_corr': [channel_avg_corr[ch] for ch in bottom_channels],
    }
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Average Loss: {avg_loss:.6f}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average Correlation: {avg_correlation:.6f}")
    print("\nTop 10 Channels by Correlation:")
    for i, ch in enumerate(top_channels):
        if channel_avg_corr[ch] > 0:  # Only show channels with valid correlation
            print(f"  Channel {ch}: {channel_avg_corr[ch]:.4f}")
    
    # Plot channel correlations
    plt.figure(figsize=(10, 6))
    valid_channels = [i for i, corr in enumerate(channel_avg_corr) if corr > 0]
    valid_corrs = [channel_avg_corr[i] for i in valid_channels]
    
    plt.bar(valid_channels, valid_corrs)
    plt.axhline(y=avg_correlation, color='r', linestyle='--', label=f'Avg: {avg_correlation:.4f}')
    plt.xlabel('Channel')
    plt.ylabel('Correlation')
    plt.title('Reconstruction Correlation by Channel')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'channel_correlations.png'))
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Average Loss: {avg_loss:.6f}\n")
        f.write(f"Average MSE: {avg_mse:.6f}\n")
        f.write(f"Average Correlation: {avg_correlation:.6f}\n")
        f.write("\nTop 10 Channels by Correlation:\n")
        for i, ch in enumerate(top_channels):
            if channel_avg_corr[ch] > 0:
                f.write(f"  Channel {ch}: {channel_avg_corr[ch]:.4f}\n")
    
    print(f"\nResults saved to {args.output_dir}")
    return results

def main():
    args = parse_args()
    
    # Load config
    config = Config_MBM_EEG()
    
    # Override with command line args if provided
    if args.data_path is not None:
        config.test_path = args.data_path
    
    print(f"Loading data from: {config.test_path}")
    print(f"Loading checkpoint from: {args.checkpoint}")
    
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
        use_nature_img_loss=config.use_nature_img_loss,
        autoencoder_path='./autoencoder_results/autoencoder_final.pth',
        projection_type='autoencoder'
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    
    # Create dataloader
    dataloader = get_dataloader(root_dir=config.test_path, batch_size=args.batch_size)
    
    # Run evaluation
    evaluate_model(model, dataloader, args, config)

if __name__ == "__main__":
    main() 

'''python evaluate_model.py   --checkpoint /home/teaching/Documents/G28_dl/Human-Gait-Representation-Learning/code_ds/final_model.pth   --data_path /home/teaching/Documents/G28_dl/cs671-p07/IMU_Dataset/Dataset/Self_supervise_data/validation   --output_dir ./evaluation_results   --batch_size 32   --num_samples 100'''