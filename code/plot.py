import torch
import matplotlib.pyplot as plt
from mae import MAEforEEG
from training import IMUAdapter
from dataset import IMUDataset
from config import Config_MBM_EEG
from torch.utils.data import DataLoader

def visualize_reconstruction(checkpoint_path, num_samples=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    config = Config_MBM_EEG()
    config.time_len = 128
    config.in_chans = 6

    # Load dataset and dataloader
    dataset = IMUDataset(
        root_dir=config.root_path,
        window_size=config.time_len,
        features=config.in_chans,
        subjects_per_batch=1,
        files_per_subject=1
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Load model and adapter
    imu_adapter = IMUAdapter(out_chans=128, out_time=512).to(device)
    model = MAEforEEG(time_len=512,
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
        use_nature_img_loss=config.use_nature_img_loss).to(device)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            batch = batch.to(device)
            if batch.dim() == 4:
                batch = batch.squeeze(0)
            if batch.dim() == 2:
                batch = batch.unsqueeze(0)

            batch = imu_adapter(batch)  # -> [B, 512, 128]
            

            loss, pred, mask = model(batch)
            recon = model.unpatchify(pred )  # [1, 128, 512]

            orig = batch.detach().cpu().squeeze().transpose(0, 1)  # [128, 512] → [512, 128]
            recon = recon.detach().cpu().squeeze().transpose(0, 1)  # [128, 512] → [512, 128]

            # Plot
            fig, axs = plt.subplots(nrows=3, figsize=(12, 6))
            for ch in range(min(3, orig.shape[1])):  # Plot first 3 channels
                axs[ch].plot(orig[:, ch], label='Original', alpha=0.7)
                axs[ch].plot(recon[:, ch], label='Reconstructed', alpha=0.7)
                axs[ch].set_title(f'Sample {i+1} - Channel {ch}')
                axs[ch].legend()
            plt.tight_layout()
            plt.show()

if __name__ == '__main__':
    checkpoint_path = r'D:\Self_supervise_data\exps\imu_pretrain\16-04-2025-21-41-09\final_model.pth'
    visualize_reconstruction(checkpoint_path, num_samples=3)