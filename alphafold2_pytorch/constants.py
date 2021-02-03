import torch

# constants

MAX_NUM_MSA = 20
NUM_AMINO_ACIDS = 21
NUM_EMBEDDS_TR = 1280 # best esm model 
DISTOGRAM_BUCKETS = 37

# default device

DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_NAME)
