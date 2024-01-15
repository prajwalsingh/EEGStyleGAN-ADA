import numpy as np
import torch
import os

np.random.seed(45)
torch.manual_seed(45)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 



base_path       = '/home/brainimage/'
train_path      = 'brain2image/dataset/eeg_imagenet40_cvpr_2017/train/'
validation_path = 'brain2image/dataset/eeg_imagenet40_cvpr_2017/val/'
test_path       = 'brain2image/dataset/eeg_imagenet40_cvpr_2017/test/'
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

# Hyper-parameters
embedding_dim  = 256
image_dim = 256
projection_dim = 256
input_channels = 1#128 # Number of EEG channels
hidden_size    = embedding_dim//2
num_layers     = 1
batch_size     = 8
test_batch_size= 1
epoch          = 4097
temperature    = 0.5