# fl_framework/components/optimizers/base_optimizer.py

import torch
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """
    Abstract base class for all optimizers in the framework.

    The core design principle of this framework is that the Optimizer is the "brain"
    that contains all the training logic. It achieves this by registering its
    logic into the framework's lifecycle via hooks.

    Every concrete optimizer must inherit from this class and implement the
    `register_hooks` method.
    """

    def __init__(self, lr: float):
        """
        Initializes the base optimizer.

        Args:
            lr (float): The learning rate.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        self.lr = lr

    def register_hooks(self):
        """
        Method for registering all necessary hooks.
        """
        pass


    @torch.no_grad()
    @abstractmethod
    # def step(self, server, aggregated_grad) -> None:
    def step(self, server, context)  -> None:

        """
        Performs the global model update based on aggregated gradients.

        This function is the "brain" of the optimizer. It is triggered by the
        Coordinator via the hook system. It accesses the server model and the
        aggregated gradient from the context and performs the update in-place.

        Args:
            server (Server): The server instance containing the global model.
            aggregated_grad (torch.Tensor): The aggregated gradients from clients.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_state(self) -> dict:
        """
        Returns the current state of the optimizer.

        Returns:
            dict: A dictionary containing the state of the optimizer.
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_state(self, state: dict):
        """
        Sets the state of the optimizer.

        Args:
            state (dict): A dictionary containing the state of the optimizer.
        """
        raise NotImplementedError