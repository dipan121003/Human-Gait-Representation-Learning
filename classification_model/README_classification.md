# Classification Model for Human Gait Analysis

This project extends the Masked Autoencoder (MAE) model for human gait classification tasks using IMU data. It uses the pre-trained MAE encoder as a feature extractor and adds a classification head on top for downstream tasks.

## Overview

The classification architecture consists of:
1. A pre-trained MAE encoder that extracts features from IMU data
2. An MLP classification head that takes these features and predicts one of 6 classes

## Data Organization

For classification, your data should be organized in a specific folder structure:

```
data_root/
├── class1/
│   ├── sample1.csv
│   ├── sample2.csv
│   └── ...
├── class2/
│   ├── sample1.csv
│   └── ...
├── class3/
│   └── ...
└── ...
```

Where each subfolder represents a class, and contains CSV files with IMU data.

## Usage

### Training the Classifier

```bash
python classification_model.py \
  --mae_checkpoint /path/to/mae_checkpoint.pth \
  --train_data /path/to/train_data \
  --val_data /path/to/val_data \
  --test_data /path/to/test_data \
  --output_dir ./classification_results
```

### Key Parameters

- `--mae_checkpoint`: Path to the pre-trained MAE model checkpoint
- `--train_data`, `--val_data`, `--test_data`: Paths to training, validation, and test data
- `--num_classes`: Number of classes (default: 6)
- `--freeze_encoder`: Whether to freeze the encoder weights (default: True)
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 1e-4)

### Full Parameter List

```
Model parameters:
  --hidden_dim          Hidden dimension of classifier (default: 256)
  --num_classes         Number of classes (default: 6)
  --dropout             Dropout probability (default: 0.3)
  --freeze_encoder      Whether to freeze encoder weights (default: True)

Training parameters:
  --batch_size          Batch size (default: 32)
  --epochs              Number of epochs (default: 30)
  --lr                  Learning rate (default: 1e-4)
  --min_lr              Minimum learning rate (default: 1e-6)
  --weight_decay        Weight decay (default: 0.01)
  --clip_grad           Gradient clipping (default: 1.0)
```

## Model Architecture

The classification model uses the MAE encoder as a feature extractor and passes the features through an MLP classification head:

1. **Feature Extractor**:
   - Uses the pre-trained MAE encoder
   - Averages the token representations (does not use CLS token)
   - Outputs a feature vector of dimension equal to the MAE embedding dimension (default: 1024)
 
2. **Classification Head**:
   - Multi-layer perceptron with:
     - Input: Feature vector from encoder (1024 dimensions)
     - Hidden layer 1: 256 dimensions
     - Hidden layer 2: 128 dimensions
     - Output: 6 class probabilities

## Results and Visualization

The training process saves:
- Training loss and validation accuracy curves
- Best model based on validation accuracy
- Label mapping

The testing process generates:
- Classification report with precision, recall, and F1-score
- Confusion matrix visualization
- Overall accuracy

All results are saved in the specified output directory.

## Example Workflow

1. Pre-train the MAE model on unlabeled IMU data:
   ```bash
   python training.py
   ```

2. Fine-tune a classifier using the pre-trained MAE:
   ```bash
   python classification_model.py \
     --mae_checkpoint /path/to/mae/checkpoint.pth \
     --train_data /path/to/labeled/train_data \
     --val_data /path/to/labeled/val_data \
     --test_data /path/to/labeled/test_data
   ```

3. Evaluate the model:
   - Review training curves and confusion matrix
   - Check classification report for per-class performance 