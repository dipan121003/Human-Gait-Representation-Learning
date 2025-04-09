# visualize_recon.py
import torch
import matplotlib.pyplot as plt
from dataset import IMUDataset
from mae import MAEforEEG
from utils import load_model
from config import Config_MBM_EEG
from training import IMUAdapter
import os

def plot_reconstruction(original, reconstructed, sample_idx=0, channels_to_plot=[0, 1, 2]):
    """
    original, reconstructed: tensors of shape [B, T, C]
    """
    original = original.detach().cpu()
    reconstructed = reconstructed.detach().cpu()

    for c in channels_to_plot:
        plt.figure(figsize=(10, 3))
        plt.plot(original[sample_idx, :, c], label='Original (Adapted Input)', linewidth=1.5)
        plt.plot(reconstructed[sample_idx, :, c], label='Reconstructed', linestyle='--', linewidth=1.5)
        plt.title(f'Channel {c} - Sample {sample_idx}')
        plt.xlabel('Time Step')
        plt.ylabel('Signal Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def main():
    config = Config_MBM_EEG()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = IMUDataset(
        root_dir=config.root_path,
        window_size=config.time_len,
        features=config.in_chans,
        subjects_per_batch=1,
        files_per_subject=4,
        transform=None
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Load IMUAdapter
    imu_adapter = IMUAdapter(out_chans=128, out_time=512).to(device)

    # Load model
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
    ).to(device)

    # Load checkpoint
    checkpoint_path = os.path.join(config.output_path, 'imu_pretrain', 'latest_run_folder', 'final_model.pth')  # <- adjust path
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Visualize 1 batch
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)               # [1, 128, 6]
            adapted_input = imu_adapter(batch)     # [1, 512, 128]
            loss, pred, _ = model(adapted_input)   # pred: [1, L, p]
            recon = model.unpatchify(pred)         # [1, 512, 128]

            # Plot
            plot_reconstruction(adapted_input, recon, sample_idx=0, channels_to_plot=[0, 1, 2])
            break  # Only visualize one batch

if __name__ == '__main__':
    main()
