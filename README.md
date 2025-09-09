# Human Gait Representation Learning

This project implements a Masked Autoencoder (MAE) for learning representations from IMU (Inertial Measurement Unit) data for human gait analysis.

## Project Structure

- `config.py`: Configuration settings for the model
- `dataset.py`: Dataset loading and preprocessing
- `mae.py`: Implementation of the Masked Autoencoder model
- `trainer.py`: Training utilities
- `training.py`: Main training script
- `utils.py`: Helper functions
- `visualize_output.py`: Visualization tool for model input vs reconstruction
- `evaluate_model.py`: Evaluation metrics and performance analysis
- `visualize_masking.py`: Visualization of the masking process

## Visualization Tools

The project includes three visualization tools to help understand the model:

### 1. Input vs Output Visualization

This tool allows you to visualize how well the model is reconstructing the original input data.

```bash
python visualize_output.py --checkpoint path/to/model/checkpoint.pth --data_path path/to/test/data --output_dir ./visualizations
```

Optional arguments:
- `--num_samples 5`: Number of samples to visualize
- `--channels 0 1 2`: Which channels to plot (defaults to first 3)
- `--device cuda`: Device to run on ('cuda' or 'cpu')

### 2. Model Evaluation

This tool provides quantitative evaluation of the model's performance.

```bash
python evaluate_model.py --checkpoint path/to/model/checkpoint.pth --data_path path/to/test/data --output_dir ./evaluation_results
```

Optional arguments:
- `--batch_size 32`: Batch size for evaluation
- `--num_samples 100`: Limit number of samples to evaluate (None = use all)
- `--device cuda`: Device to run on ('cuda' or 'cpu')

### 3. Masking Visualization

This tool helps visualize the masking and reconstruction process to understand what parts of the input the model is focusing on.

```bash
python visualize_masking.py --checkpoint path/to/model/checkpoint.pth --data_path path/to/test/data --output_dir ./masking_visualizations
```

Optional arguments:
- `--num_samples 3`: Number of samples to visualize
- `--channels 0 1 2`: Which channels to plot (defaults to first 3)
- `--mask_ratio 0.75`: Ratio of patches to mask
- `--device cuda`: Device to run on ('cuda' or 'cpu')

## Training the Model

To train the model from scratch:

```bash
python training.py
```

## Using a Pre-trained Model

If you have a pre-trained model, you can evaluate it or visualize its outputs using the provided tools.

Example:

```bash
# Evaluate the model
python evaluate_model.py --checkpoint /path/to/checkpoint.pth --data_path /path/to/test/data

# Visualize reconstructions
python visualize_output.py --checkpoint /path/to/checkpoint.pth --data_path /path/to/test/data
```

## Project Details

This project uses a Masked Autoencoder architecture for self-supervised representation learning from time-series IMU data. The model is trained by:

1. Masking random parts of the input signal (default: 75% masked)
2. Encoding the visible portions with a Transformer encoder
3. Reconstructing the full signal with a Transformer decoder
4. Training on the reconstruction loss

The learned representations can be used for various downstream tasks related to human gait analysis. 