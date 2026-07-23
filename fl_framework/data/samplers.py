# fl_framework/data/samplers.py

from abc import ABC, abstractmethod
from typing import Any, Iterator
import torch
from torch.utils.data import Dataset, DataLoader

class BaseSampler(ABC):
    """
    Abstract base class for client-side data samplers.

    A sampler is responsible for providing data samples (or batches)
    to a client during its local training process.
    """
    @abstractmethod
    def get_sample(self) -> Any:
        """
        Returns a single sample or a mini-batch of data.
        """
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

class MiniBatchSampler(BaseSampler):
    """
    A sampler that provides mini-batches of data from a given dataset.

    It internally uses a PyTorch DataLoader and automatically handles
    reshuffling and looping over the data when an epoch ends.
    """
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True):
        """
        Initializes the MiniBatchSampler.

        Args:
            dataset (Dataset): The client's local dataset partition.
            batch_size (int): The size of each mini-batch.
            shuffle (bool): Whether to shuffle the data at the beginning of each epoch.
        """
        if not isinstance(dataset, Dataset) or len(dataset) == 0:
            raise ValueError("Dataset must be a non-empty torch.utils.data.Dataset")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Keep tail batches so small or highly skewed client partitions remain usable.
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=False
        )
        self._iterator: Iterator = iter(self.dataloader)

    def get_sample(self) -> Any:
        """
        Provides the next mini-batch of data.

        If the current epoch is exhausted, it transparently starts a new one.

        Returns:
            Any: A mini-batch of data, typically a tuple of (features, labels).
        """
        try:
            data = next(self._iterator)
        except StopIteration:
            # Epoch finished, create a new iterator to loop over the data again
            self._iterator = iter(self.dataloader)
            data = next(self._iterator)
        
        return data
    
    def __len__(self):
        return len(self.dataloader)
    