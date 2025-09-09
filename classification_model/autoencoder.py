import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from Foundation_model_training.dataset import get_dataloader
import time

class IMUAutoencoder(nn.Module):
    def __init__(self, input_channels=6, hidden_channels=128, time_len=212):
        super().__init__()
        
        # Encoder - transforms 6 channels to 128 channels
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_channels),
        )
        
        # Decoder - transforms 128 channels back to 6 channels
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, input_channels, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        # x shape: [B, T, C] -> need to permute to [B, C, T]
        if x.shape[1] != 6 and x.shape[2] == 6:
            x = x.permute(0, 2, 1)
            
        # Encode the input
        encoded = self.encoder(x)
        
        # Decode the encoded representation
        decoded = self.decoder(encoded)
        
        return encoded, decoded

@torch.no_grad()
def validate(model, data_loader, criterion, device):
    """
    Validate the autoencoder model on a validation dataset
    """
    model.eval()
    val_losses = []
    
    for batch_idx, data in enumerate(data_loader):
        # Move data to device and ensure proper shape
        data = data.to(device)
        
        if data.ndim == 3 and data.shape[2] == 6:  # [B, T, C]
            data = data.permute(0, 2, 1)  # -> [B, C, T]
        
        # Forward pass
        _, reconstructed = model(data)
        
        # Calculate loss
        loss = criterion(reconstructed, data)
        val_losses.append(loss.item())
    
    # Return average validation loss
    return sum(val_losses) / len(val_losses)

def train_autoencoder(model, train_loader, val_loader, device, num_epochs=50, lr=1e-3, output_dir="./autoencoder_results", val_freq=5):
    """
    Train the autoencoder model with validation every val_freq epochs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Training loop
    train_losses = []
    val_losses = []
    val_epochs = []
    best_val_loss = float('inf')
    
    # Training visualization
    plt.figure(figsize=(10, 6))
    plt.ion()  # Interactive mode ON
    
    model.to(device)
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_losses = []
        
        for batch_idx, data in enumerate(train_loader):
            # Move data to device and ensure proper shape
            data = data.to(device)
            
            if data.ndim == 3 and data.shape[2] == 6:  # [B, T, C]
                data = data.permute(0, 2, 1)  # -> [B, C, T]
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            _, reconstructed = model(data)
            
            # Calculate loss
            loss = criterion(reconstructed, data)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Record loss
            loss_value = loss.item()
            epoch_losses.append(loss_value)
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss_value:.6f}")
        
        # Update learning rate
        scheduler.step()
        
        # Compute average epoch loss for training
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_train_loss)
        
        # Validation phase (only every val_freq epochs)
        run_validation = (epoch + 1) % val_freq == 0 or epoch == num_epochs - 1
        
        if run_validation:
            val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_epochs.append(epoch + 1)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(output_dir, 'autoencoder_best.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with validation loss: {best_val_loss:.6f}")
            
            # Visualize reconstruction at validation epochs
            visualize_reconstruction(model, val_loader, device, epoch+1, output_dir)
        
        # Update loss plot
        plt.clf()
        plt.plot(range(1, epoch+2), train_losses, 'b-', label='Training Loss')
        if val_epochs:
            plt.plot(val_epochs, val_losses, 'ro-', label='Validation Loss')
        plt.title('Autoencoder Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.draw()
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
        plt.pause(0.1)
        
        # Print epoch summary
        log_message = f"Epoch {epoch+1}/{num_epochs} complete - Train Loss: {avg_train_loss:.6f}"
        if run_validation:
            log_message += f", Val Loss: {val_loss:.6f}"
        log_message += f", Time: {(time.time() - start_time)/60:.2f} min"
        print(log_message)
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(output_dir, f'autoencoder_epoch_{epoch+1}.pth')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
            }
            if run_validation:
                checkpoint['val_loss'] = val_loss
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
    
    plt.ioff()  # Turn off interactive mode
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'autoencoder_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete! Final model saved at {final_model_path}")
    
    return model

def visualize_reconstruction(model, data_loader, device, epoch, output_dir):
    """
    Visualize original and reconstructed signals
    """
    model.eval()
    
    # Get a batch of data
    batch = next(iter(data_loader))
    batch = batch.to(device)
    
    # Ensure proper shape
    if batch.ndim == 3 and batch.shape[2] == 6:  # [B, T, C]
        batch = batch.permute(0, 2, 1)  # -> [B, C, T]
    
    with torch.no_grad():
        # Get reconstruction
        _, reconstruction = model(batch)
    
    # Move tensors to CPU for visualization
    batch = batch.cpu().numpy()
    reconstruction = reconstruction.cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(6, 2, figsize=(15, 12))
    fig.suptitle(f'Original vs Reconstructed Signals (Epoch {epoch})', fontsize=16)
    
    # Plot each channel for a single sample
    sample_idx = 0
    for i in range(6):
        # Original signal
        axes[i, 0].plot(batch[sample_idx, i, :])
        axes[i, 0].set_title(f'Original Ch{i+1}')
        axes[i, 0].grid(True)
        
        # Reconstructed signal
        axes[i, 1].plot(reconstruction[sample_idx, i, :], 'r')
        axes[i, 1].set_title(f'Reconstructed Ch{i+1}')
        axes[i, 1].grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'reconstruction_epoch_{epoch}.png'))
    plt.close(fig)

def get_pretrained_encoder(model_path, input_channels=6, hidden_channels=128, time_len=212, device='cuda'):
    """
    Load a pretrained autoencoder and return its encoder part only
    """
    # Create a full autoencoder model
    autoencoder = IMUAutoencoder(input_channels, hidden_channels, time_len)
    
    # Load the pretrained weights
    autoencoder.load_state_dict(torch.load(model_path, map_location=device))
    
    # Return only the encoder part
    return autoencoder.encoder

if __name__ == "__main__":
    # Configuration
    batch_size = 64
    num_epochs = 25
    learning_rate = 1e-3
    output_dir = "./autoencoder_results_2nd"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_frequency = 5  # Validate every 5 epochs
    
    # Data paths
    train_dir = "/home/teaching/Documents/G28_dl/cs671-p07/IMU_Dataset/Dataset/Self_supervise_data/train"
    val_dir = "/home/teaching/Documents/G28_dl/cs671-p07/IMU_Dataset/Dataset/Self_supervise_data/validation"
    
    # Load data
    print("Loading data...")
    train_loader = get_dataloader(train_dir, batch_size=batch_size)
    val_loader = get_dataloader(val_dir, batch_size=batch_size)
    
    # Create autoencoder model
    print("Creating autoencoder model...")
    model = IMUAutoencoder()
    
    # Train model
    print("Starting training...")
    trained_model = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        lr=learning_rate,
        output_dir=output_dir,
        val_freq=validation_frequency
    )
    
    print("Training complete!") 