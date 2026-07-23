# fl_framework/components/aggregators/base_aggregator.py

from abc import ABC, abstractmethod
from typing import List
import torch

class BaseAggregator(ABC):
    """
    Abstract base class for all aggregation algorithms.

    An Aggregator is responsible for combining the gradients (or other updates)
    from multiple clients into a single, aggregated gradient. This component
    works in tandem with the Optimizer but handles a distinct responsibility.

    This design allows for easy swapping of aggregation strategies (e.g., mean,
    median, trimmed mean) without altering the Server or Coordinator logic.
    """

    @torch.no_grad()
    @abstractmethod
    def aggregate(
        self,
        all_gradients: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Aggregates gradients from multiple clients.

        Args:
            all_gradients (List[List[torch.Tensor]]): A list where each element
                is the list of gradient tensors from a single client.
                Example: [[client1_grad1, client1_grad2], [client2_grad1, client2_grad2]]

        Returns:
            List[torch.Tensor]: A single list of tensors representing the
                aggregated gradient.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_state(self) -> dict:
        """
        Returns the current state of the aggregator.

        Returns:
            dict: A dictionary containing the state of the aggregator.
        """
        raise NotImplementedError

    @abstractmethod
    def set_state(self, state: dict):
        """
        Sets the state of the aggregator.

        Args:
            state (dict): A dictionary containing the state to set.
        """
        raise NotImplementedError