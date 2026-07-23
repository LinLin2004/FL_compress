# fl_framework/data/base_dataset.py

from abc import ABC, abstractmethod
from typing import List, Any, Tuple
from torch.utils.data import Dataset

class BaseFLDataset(Dataset, ABC):
    """
    Abstract base class for all datasets used in this framework.

    It inherits from `torch.utils.data.Dataset` and adds an abstract property
    `targets`, which is crucial for performing efficient label-based
    Non-IID data partitioning.

    Any custom dataset created for this framework must inherit from this class
    and implement all its abstract methods.
    """

    @property
    @abstractmethod
    def targets(self) -> List[int]:
        """
        Returns a list of all labels in the dataset.

        This is a mandatory property for enabling efficient Non-IID partitioning
        without iterating through the entire dataset.

        Returns:
            List[int]: A list where the i-th element is the label of the
                       i-th data point.
        """
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        """Returns the number of unique classes in the dataset."""
        return len(set(self.targets))

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Gets the data and label at the specified index.

        Args:
            index (int): The index of the data point.

        Returns:
            Tuple[Any, Any]: A tuple containing (data, label).
        """
        raise NotImplementedError
