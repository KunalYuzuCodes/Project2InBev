import torch

CUDA_VISIBLE_DEVICES = "0"  # Use first GPU
BATCH_SIZE = 16  # Adjust based on your GPU memory
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
EPOCHS = 3

# CUDA settings
CUDA_LAUNCH_BLOCKING = "1"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
