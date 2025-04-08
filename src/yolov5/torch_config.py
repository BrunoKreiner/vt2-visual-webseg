import os
import warnings

# Configure CUDA environment variables
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Import torch after setting environment variables
import torch
print(torch.__version__)

# Basic PyTorch configurations
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Tell PyTorch to be lenient about operations that aren't deterministic
torch.use_deterministic_algorithms(False)

# Suppress warnings about deprecated features and non-deterministic operations
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)