import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import pandas as pd

# Import local modules
from Foundation_model_training.config import Config_MBM_EEG
from Foundation_model_training.dataset import get_dataloader, IMUDataset
from Foundation_model_training.mae import MAEforEEG
from Foundation_model_training.utils import load_model

class EncoderExtractor(nn.Module):
    """Module that extracts features from the MAE encoder only."""
    def __init__(self, mae_model, use_cls_token=False):
        super().__init__()
        self.mae = mae_model
        self.use_cls_token = use_cls_token
        
    def forward(self, x):
        # Get the latent representation from encoder only
        with torch.no_grad():
            latent, mask, ids_restore ,_= self.mae.forward_encoder(x.permute(0, 2, 1), mask_ratio=0)
        
        # Remove CLS token if present, then flatten all tokens for each sample
        if latent.shape[1] > 1:  # If there's a CLS token
            features = latent[:, 1:, :].reshape(latent.size(0), -1)  # Flatten all non-CLS tokens
        else:
            features = latent.reshape(latent.size(0), -1)  # Flatten all tokens if no CLS
        return features

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_classes=6, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.mlp(x)


class MAEClassifier(nn.Module):
    """Full classifier with MAE encoder + classification head."""
    def __init__(self, mae_model, hidden_dim=256, num_classes=6, dropout=0.3, freeze_encoder=True):
        super().__init__()
        
        # MAE feature extractor
        self.feature_extractor = EncoderExtractor(mae_model, use_cls_token=False)
        
        # Determine input dimension from encoder output (flattened tokens)
        self.encoder_dim = mae_model.num_patches * mae_model.embed_dim
        
        # Classification head
        self.classifier = ClassificationHead(
            input_dim=self.encoder_dim, 
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Freeze the encoder if specified
        if freeze_encoder:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits

# Classification dataset wrapper
class IMUClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, label_map=None, window_size=212,training=True):
        self.training = training
        self.window_size = window_size  # <-- ADD THIS LINE
        self.dataset = IMUDataset(root_dir, window_size=window_size)
        
        # Create a label map if not provided (useful for validation/test sets)
        if label_map is None:
            # Assuming subdirectories represent classes
            self.classes = sorted([d for d in os.listdir(root_dir) 
                                if os.path.isdir(os.path.join(root_dir, d))])
            self.label_map = {cls: i for i, cls in enumerate(self.classes)}
        else:
            self.label_map = label_map
            self.classes = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]

        # Create (data_path, label) pairs
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for file in os.listdir(class_dir):
                if file.lower().endswith('.csv'):
                    self.samples.append((
                        os.path.join(class_dir, file),
                        self.label_map[class_name]
                    ))

    
    def __len__(self):
        return len(self.samples)
    def jitter(self, data, sigma=0.01):
        noise = np.random.normal(loc=0.0, scale=sigma, size=data.shape)
        return data + noise

    def scale(self, data, sigma=0.1):
        scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(1, data.shape[1]))
        return data * scaling_factor

    def permute(self, data, n_segments=4):
        seg_len = data.shape[0] // n_segments
        segments = [data[i*seg_len:(i+1)*seg_len] for i in range(n_segments)]
        np.random.shuffle(segments)
        return np.concatenate(segments, axis=0)

    def __getitem__(self, idx):
        '''file_path, label = self.samples[idx]

        # Load and clean the data
        df = pd.read_csv(file_path, header=None)
        data = df.apply(pd.to_numeric, errors='coerce').values.astype(np.float32)

        # Normalize
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        std[std == 0] = 1.0
        data = (data - mean) / std
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Sample or reroll to fixed window
        if data.shape[0] >= self.window_size:
            start_idx = np.random.randint(0, data.shape[0] - self.window_size + 1)
            window = data[start_idx:start_idx + self.window_size]
        else:
            # Loop data to match window size (reroll)
            repeat_times = (self.window_size // data.shape[0]) + 1
            rolled_data = np.tile(data, (repeat_times, 1))
            window = rolled_data[:self.window_size]

        return torch.tensor(window, dtype=torch.float32), label'''
        file_path, label = self.samples[idx]

        # Load and clean the data
        df = pd.read_csv(file_path, header=None)
        data = df.apply(pd.to_numeric, errors='coerce').values.astype(np.float32)

        # Normalize
        # mean = np.nanmean(data, axis=0)
        # std = np.nanstd(data, axis=0)
        # std[std == 0] = 1.0
        # data = (data - mean) / std
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Reroll if too short
        if data.shape[0] < self.window_size:
            repeat_times = (self.window_size // data.shape[0]) + 1
            data = np.tile(data, (repeat_times, 1))
        data = data[:self.window_size]

        # === Augmentations ===
        if self.training:  # Flag to control augmentation only during training
            data = self.jitter(data, sigma=0.01)
            data = self.scale(data, sigma=0.1)
            data = self.permute(data, n_segments=4)

        return torch.tensor(data, dtype=torch.float32), label



def get_classification_dataloader(root_dir, batch_size=32, label_map=None, num_workers=4,training=False):
    dataset = IMUClassificationDataset(root_dir, label_map=label_map,training=training)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    ), dataset.label_map

from torch.nn.functional import cross_entropy

def focal_loss(logits, targets, alpha=1.0, gamma=2.0, reduction='mean'):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # pt = softmax prob of the true class
    loss = alpha * (1 - pt) ** gamma * ce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def train_classifier(model, train_loader, val_loader, args):
    """Train the classifier."""
    device = torch.device(args.device)
    model = model.to(device)
    # Replace your criterion declaration inside train_classifier()
    def criterion(logits, targets):
        return focal_loss(logits, targets, alpha=1.0, gamma=2.0)

    
    #criterion = nn.CrossEntropyLoss()

    
    # Separate parameters that require gradients for optimizer
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_update, lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    
    # Training loop
    best_accuracy = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(params_to_update, args.clip_grad)
                
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % args.log_interval == 0:
                print(f"Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                      f" ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}, Average training loss: {avg_train_loss:.6f}")
        
        # Validation
        val_accuracy, val_loss = evaluate(model, val_loader, criterion, device)
        val_accuracies.append(val_accuracy)
        
        print(f"Validation accuracy: {val_accuracy:.4f}, loss: {val_loss:.6f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            if args.output_dir:
                save_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': val_accuracy,
                }, save_path)
                print(f"Saved best model with accuracy {val_accuracy:.4f} to {save_path}")
        
        # Save checkpoint
        if args.output_dir and (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_accuracy,
            }, save_path)
    
    # Plot training curves
    if args.output_dir:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies)
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
        plt.close()
    
    return model, best_accuracy

def evaluate(model, data_loader, criterion, device):
    """Evaluate the classifier."""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            val_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    val_loss /= len(data_loader)
    accuracy = correct / total
    
    return accuracy, val_loss

from collections import Counter

def test_classifier(model, test_dir, args, label_map):
    """Test the classifier using voting-based inference per file."""
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    window_size = 212  # Match the model input size
    all_preds = []
    all_targets = []
    all_files = []

    print("Testing with voting per CSV file...")
    with torch.no_grad():
        for class_dir in os.listdir(test_dir):
            class_path = os.path.join(test_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            true_label = label_map[class_dir]

            for fname in os.listdir(class_path):
                if not fname.endswith(".csv"):
                    continue
                file_path = os.path.join(class_path, fname)
                df = pd.read_csv(file_path, header=None)
                data = df.apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                if data.shape[0] < window_size:
                    repeat_times = (window_size // data.shape[0]) + 1
                    data = np.tile(data, (repeat_times, 1))

                num_windows = data.shape[0] // window_size
                data = data[:num_windows * window_size]
                windows = data.reshape(num_windows, window_size, -1)

                votes = []
                for window in windows:
                    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
                    output = model(x)
                    probs = F.softmax(output, dim=1).squeeze()
                    votes.append(probs.cpu().numpy())
                summed_probs = np.sum(votes, axis=0)
                majority_vote = np.argmax(summed_probs)

                all_preds.append(majority_vote)
                all_targets.append(true_label)
                all_files.append(fname)

    accuracy = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, digits=4,zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Voting-Based File-Level Test Accuracy: {accuracy:.6f}\n\n")
        f.write(report)

    df_results = pd.DataFrame({
        "file": all_files,
        "true_label": all_targets,
        "predicted_label": all_preds
    })
    df_results.to_csv(os.path.join(args.output_dir, "voting_test_results.csv"), index=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Voting-Based Confusion Matrix')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    plt.close()

    print(f"Voting-based testing complete. Accuracy: {accuracy:.4f}")
    print(f"Results saved to {args.output_dir}")

    return accuracy


def parse_args():
    parser = argparse.ArgumentParser('MAE Classifier', add_help=False)
    parser.add_argument('--mae_checkpoint', type=str, required=True, help='path to MAE model checkpoint')
    parser.add_argument('--train_data', type=str, required=True, help='path to training data')
    parser.add_argument('--val_data', type=str, required=True, help='path to validation data')
    parser.add_argument('--test_data', type=str, default=None, help='path to test data (optional)')
    parser.add_argument('--output_dir', type=str, default='./classifier_results', help='output directory')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension of classifier')
    parser.add_argument('--num_classes', type=int, default=6, help='number of classes')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout probability')
    parser.add_argument('--freeze_encoder', type=bool, default=True, help='whether to freeze encoder weights')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    parser.add_argument('--save_interval', type=int, default=5, help='save interval')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='device to run on')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config for MAE model
    config = Config_MBM_EEG()
    
    # Load MAE model
    print(f"Loading MAE model from checkpoint: {args.mae_checkpoint}")
    mae_model = MAEforEEG(
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
        autoencoder_path='./autoencoder_results/autoencoder_final.pth'
        
    )
    # projection_type='autoencoder'
    
    # Load checkpoint
    device = torch.device(args.device)
    checkpoint = torch.load(args.mae_checkpoint, map_location=device)
    if 'model' in checkpoint:
        mae_model.load_state_dict(checkpoint['model'], strict=False)
    else:
        mae_model.load_state_dict(checkpoint, strict=False)
    
    mae_model = mae_model.to(device)
    
    # Create classifier model
    classifier = MAEClassifier(
        mae_model=mae_model,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder
    )
    
    # Load data
    print(f"Loading training data from: {args.train_data}")
    train_loader, label_map = get_classification_dataloader(
        args.train_data, batch_size=args.batch_size,training=False
    )
    
    print(f"Loading validation data from: {args.val_data}")
    val_loader, _ = get_classification_dataloader(
        args.val_data, batch_size=args.batch_size, label_map=label_map,training=False
    )
    
    # Save label mapping
    with open(os.path.join(args.output_dir, 'label_map.txt'), 'w') as f:
        for class_name, idx in sorted(label_map.items(), key=lambda x: x[1]):
            f.write(f"{idx}: {class_name}\n")
    
    # Train classifier
    print("Starting training...")
    classifier, best_acc = train_classifier(classifier, train_loader, val_loader, args)
    
    # Test if test data is provided
    if args.test_data:
        print("Testing best model...")
        # Load best model
        best_model_path = os.path.join(args.output_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            classifier.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']} with accuracy {checkpoint['accuracy']:.4f}")

        # Run voting-based testing
        test_classifier(classifier, args.test_data, args, label_map)

def predict_single_sample(model, data, device):
    """Make prediction for a single sample."""
    model.eval()
    with torch.no_grad():
        data = data.unsqueeze(0).to(device)  # Add batch dimension
        output = model(data)
        probs = F.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        
    return pred, probs.squeeze().cpu().numpy()

if __name__ == "__main__":
    main() 
''' 
python classification_model.py   --mae_checkpoint /home/teaching/Documents/G28_dl/cs671-p07/IMU_Dataset/Dataset/Self_supervise_data/exps_ds/imu_pretrain_autoencoder/15-05-2025-17-39-25/final_model.pth  --train_data /home/teaching/Documents/G28_dl/cs671-p07/IMU_Dataset/Dataset/Downstream_split/train   --val_data /home/teaching/Documents/G28_dl/cs671-p07/IMU_Dataset/Dataset/Downstream_split/val   --test_data /home/teaching/Documents/G28_dl/cs671-p07/IMU_Dataset/Dataset/Downstream_split/test   --output_dir ./classifier_results   --batch_size 32   --epochs 50   --lr 1e-3   --num_classes 10  --hidden_dim 512  --device cuda

'''