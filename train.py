import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm
import os
from models.ModelPrediction import ModelPrediction

# Print model size and shape
# Size (MB): 11.200143
# Input: torch.Size([batch_size, channels, signal_length])
# Output: torch.Size([batch_size, output_size])