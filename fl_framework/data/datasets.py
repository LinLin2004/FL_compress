# fl_framework/data/datasets.py

from typing import List, Any, Tuple
from torchvision import datasets, transforms
from .base_dataset import BaseFLDataset
import re
import torch
from torch.utils.data import random_split

class MNISTDataset(BaseFLDataset):
    """
    An adapter for the torchvision MNIST dataset to make it compatible
    with the BaseFLDataset interface.
    """
    def __init__(self, data_dir: str = './data', train: bool = True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = datasets.MNIST(
            data_dir, train=train, download=True, transform=transform
        )

    @property
    def targets(self) -> List[int]:
        return self.dataset.targets.tolist()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return (index,) + self.dataset[index]

class CIFAR10Dataset(BaseFLDataset):
    """
    An adapter for the torchvision CIFAR-10 dataset to make it compatible
    with the BaseFLDataset interface.
    """
    def __init__(self, data_dir: str = './data', train: bool = True):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465] , std=[0.2023, 0.1994, 0.2010]), 
            transforms.RandomErasing(),
        ])
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465] , std=[0.2023, 0.1994, 0.2010]), 
        ])
        self.dataset = datasets.CIFAR10(
            data_dir, train=train, download=True, transform=transform_train if train else transform
        )

    @property
    def targets(self) -> List[int]:
        return self.dataset.targets

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        out  = (index,) + self.dataset[index]
        return out

class IJCNN1Dataset(BaseFLDataset):
    def __init__(self, data_file, set_size=49990, feature_size=22, train: bool = True):
        """
        Initializes the IJCNN1 dataset.

        Args:
            data_file (str): Path to the IJCNN1 dataset file.
            train (bool): Whether to use the training split (80%).
        """
        full_data = torch.zeros((set_size, feature_size), dtype=torch.float64)
        full_labels = torch.zeros(set_size, dtype=torch.float64)
        with open(data_file, 'r') as f:
            for (line, vector) in enumerate(f):
                cat, data = vector.split(' ', 1)
                full_labels[line] = 1 if cat == '1' else 0
                for piece in data.strip().split(' '):
                    match = re.search(r'(\S+):(\S+)', piece)
                    feature = int(match.group(1)) - 1
                    value = float(match.group(2))
                    if feature < feature_size:
                        full_data[line][feature] = value

        train_size = int(0.8 * set_size)
        if train:
            self.data = full_data[:train_size]
            self.labels = full_labels[:train_size]
        else:
            self.data = full_data[train_size:]
            self.labels = full_labels[train_size:]

    @property
    def targets(self) -> List[int]:
        return self.labels.tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return index, self.data[index], self.labels[index]

class CovTypeDataset(BaseFLDataset):
    def __init__(self, data_file, set_size=581012, feature_size=54, train: bool = True):
        """
        Initializes the CovType dataset.

        Args:
            data_file (str): Path to the CovType dataset file.
            train (bool): Whether to use the training split (80%).
        """
        full_data = torch.zeros((set_size, feature_size), dtype=torch.float64)
        full_labels = torch.zeros(set_size, dtype=torch.float64)
        with open(data_file, 'r') as f:
            for (line, vector) in enumerate(f):
                cat, data = vector.split(' ', 1)
                full_labels[line] = 1 if cat == '1' else 0
                for piece in data.strip().split(' '):
                    match = re.search(r'(\S+):(\S+)', piece)
                    feature = int(match.group(1)) - 1
                    value = float(match.group(2))
                    if feature < feature_size:
                        full_data[line][feature] = value

        train_size = int(0.8 * set_size)
        if train:
            self.data = full_data[:train_size]
            self.labels = full_labels[:train_size]
        else:
            self.data = full_data[train_size:]
            self.labels = full_labels[train_size:]

    @property
    def targets(self) -> List[int]:
        return self.labels.tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return index, self.data[index], self.labels[index]
