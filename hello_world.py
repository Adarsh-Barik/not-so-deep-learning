import torch
import torchvision

# for data loading
from torchvision import transforms
from torchvision.datasets import ImageFolder

# for batch size
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

# for base model, base class
import torch.nn as nn
import torch.nn.functional as F

# for plots
from matplotlib import pyplot as plt


print("Hello Not-So-Deep Learning!")

