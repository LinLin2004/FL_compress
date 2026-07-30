# fl_framework/data/datasets.py

from typing import List, Any, Tuple
from torchvision import datasets, transforms
from .base_dataset import BaseFLDataset
import re
import torch
from torch.utils.data import random_split
import json
import os
import numpy as np
from PIL import Image

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


class FedMnistDataset(BaseFLDataset):
    """
    Federated MNIST dataset loaded from LEAF-format JSON files.

    In the LEAF benchmark, FEMNIST data is partitioned by writer, where each
    writer corresponds to a natural client. The JSON files contain:
      - 'num_users': number of writers
      - 'users': list of writer IDs
      - 'num_samples': number of samples per writer
      - 'user_data': dict mapping writer ID -> {'x': [images], 'y': [labels]}

    Each image 'x' is a flattened 28x28 grayscale array (784 values),
    and 'y' is an integer label in [0, 9].

    This dataset class loads all writers' data and concatenates them into a
    single dataset, preserving the writer-to-sample mapping via the
    `client_indices` property for federated partitioning by writer.

    Args:
        data_dir (str): Directory containing train.json and test.json.
        train (bool): Whether to load the training or test split.
    """

    def __init__(self, data_dir: str = './data', train: bool = True):
        json_file = os.path.join(data_dir, 'train.json' if train else 'test.json')

        if not os.path.exists(json_file):
            raise FileNotFoundError(
                f"FedMnist JSON file not found at {json_file}. "
                f"Please download the LEAF FEMNIST data and place it in {data_dir}. "
                f"Expected files: train.json, test.json"
            )

        with open(json_file, 'r') as f:
            data = json.load(f)

        users = data.get('users', [])
        user_data = data.get('user_data', {})

        all_images = []
        all_labels = []
        # Mapping from client (writer) index to sample indices in the concatenated dataset
        self._client_indices: List[List[int]] = []
        current_idx = 0

        for user_id in users:
            user_samples = user_data[user_id]
            x_list = user_samples['x']  # list of 784-length lists
            y_list = user_samples['y']  # list of int labels

            num_samples = len(x_list)
            self._client_indices.append(
                list(range(current_idx, current_idx + num_samples))
            )

            for x, y in zip(x_list, y_list):
                # Convert flattened 784-dim array to 1x28x28 tensor
                img_array = np.array(x, dtype=np.float32).reshape(28, 28)
                all_images.append(img_array)
                all_labels.append(int(y))

            current_idx += num_samples

        # Stack into tensors
        self._data = torch.tensor(np.stack(all_images), dtype=torch.float32)
        # Shape: (N, 28, 28) -> add channel dim -> (N, 1, 28, 28)
        self._data = self._data.unsqueeze(1)
        self._targets = all_labels

        # Apply normalization (same as standard MNIST)
        self._transform = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self._num_writers = len(users)
        print(f"FedMnistDataset loaded: {len(self)} samples from {self._num_writers} writers "
              f"({'train' if train else 'test'} split)")

    @property
    def targets(self) -> List[int]:
        return self._targets

    @property
    def client_indices(self) -> List[List[int]]:
        """
        Returns the mapping from writer (client) index to sample indices.

        This enables natural federated partitioning by writer, preserving
        the inherent Non-IID distribution of the LEAF benchmark.

        Returns:
            List[List[int]]: A list where element i contains the sample
            indices belonging to writer i.
        """
        return self._client_indices

    @property
    def num_writers(self) -> int:
        """Returns the number of writers (natural clients) in the dataset."""
        return self._num_writers

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = self._transform(self._data[index])
        target = self._targets[index]
        return index, data, target
