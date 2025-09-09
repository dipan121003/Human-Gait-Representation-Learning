import os
import numpy as np

class Config_MAE_fMRI: # back compatibility
    pass
class Config_MBM_finetune: # back compatibility
    pass 

class Config_MBM_EEG(Config_MAE_fMRI):
    # configs for fmri_pretrain.py / EEG pretraining
    def __init__(self):
        # Training Parameters
        self.lr = 2.5e-4
        self.min_lr = 0.
        self.weight_decay = 0.05
        self.num_epoch = 60
        self.warmup_epochs = 10
        self.batch_size = 1 # Adjust this as needed
        self.clip_grad = 0.8
        self.steps_per_epoch=128
        # Model Parameters
        self.mask_ratio = 0.75  # Updated for EEG/IMU data
        self.patch_size = 4
        self.embed_dim = 1024
        self.decoder_embed_dim = 512
        self.depth = 24
        self.num_heads = 16
        self.decoder_num_heads = 16
        self.mlp_ratio = 1.0

        # Project setting
        self.root_path = r"/home/teaching/Documents/G28_dl/cs671-p07/IMU_Dataset/Dataset/Self_supervise_data/train"
        self.output_path = r"/home/teaching/Documents/G28_dl/cs671-p07/IMU_Dataset/Dataset/Self_supervise_data/exps_ds"
        self.val_path = r"/home/teaching/Documents/G28_dl/cs671-p07/IMU_Dataset/Dataset/Self_supervise_data/validation"
        self.test_path=r"/home/teaching/Documents/G28_dl/cs671-p07/IMU_Dataset/Dataset/Self_supervise_data/test"
        self.autoencoder_path='/home/teaching/Documents/G28_dl/Human-Gait-Representation-Learning/code_ds/autoencoder_results_2nd/autoencoder_final.pth'
        self.train_dir = None  # For command line overrides
        self.val_dir = None    # For command line overrides
        self.freeze_encoder = False  # Whether to freeze autoencoder encoder weights
        self.seed = 21
        self.roi = 'VC'
        self.aug_times = 1
        self.num_sub_limit = None
        self.include_hcp = True
        self.include_kam = True
        self.accum_iter = 1

        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None
        self.focus_rate = 0.6

        # Distributed training
        self.local_rank = 0

        # EEG/IMU-specific parameters (add these for clarity)
        self.time_len = 212  # Number of time steps per sample
        self.in_chans = 6    # Number of features (channels) per time step

        # Optionally, you can add a data subdirectory here:
        self.data_subdir = 'datasets/processed_data_2'