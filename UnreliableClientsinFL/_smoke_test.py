import os
import torch
import torch.nn.functional as F
from torch import autograd
from tensorboardX import SummaryWriter
from scipy import stats
import math
from torchvision import datasets, transforms

from sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, server_test
from Update import LocalFSVGRUpdate, LocalUpdate, ServerUpdate, LocalFedProxUpdate
from FedNets import MLP1, CNNMnist, CNNCifar
from averaging import average_weights

print("All imports OK")

dataset_train = datasets.FashionMNIST('../data/fashionmnist/', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ]))
print("Dataset downloaded, size:", len(dataset_train))
print("CUDA available:", torch.cuda.is_available())
