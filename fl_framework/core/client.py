# fl_framework/core/client.py

from abc import ABC, abstractmethod
from typing import List, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import Subset

# Import dependencies from our framework
from ..components.attacks.base_attack import BaseAttack
from ..data.samplers import BaseSampler
from .hooks import Context

class BaseClient(ABC):
    """
    Abstract base class for all clients in the federated learning system.

    A client is a stateful entity that holds a local data partition, a local
    copy of the model, and is responsible for performing computations on its data.
    It acts as a "thin" component, primarily executing tasks like gradient
    computation as directed by the Coordinator.

    Attributes:
        client_id (int): A unique identifier for the client.
        data_partition (Subset): The client's slice of the total dataset.
        sampler (BaseSampler): The sampler used to generate mini-batches from the data partition.
        model (Optional[nn.Module]): A local copy of the model. It's initialized to None
                                      and set by the framework.
        device (Optional[torch.device]): The compute device (CPU or GPU) assigned to this
                                         client. It's initialized to None and set by the
                                         ResourceManager.
    """
    def __init__(self, client_id: int, data_partition: Optional[Subset] = None, sampler: Optional[BaseSampler] = None):
        self.client_id: int = client_id
        self.data_partition: Optional[Subset] = data_partition
        self.sampler: Optional[BaseSampler] = sampler
        self.model: Optional[nn.Module] = None
        self.loss: Optional[Callable] = None
        self.device: Optional[torch.device] = None
        self.other_state: dict = {}  # For custom attributes that may be added later

    def set_device(self, device: torch.device):
        """
        Assigns the model and compute device to the client.
        This is typically called by the framework before training begins.
        """
        self.device = device
        self.model.to(self.device)

    def update_model(self, new_state_dict: dict):
        """
        Updates the client's local model with a new state dictionary from the server.
        """
        if self.model:
            self.model.load_state_dict(new_state_dict)
    
    @abstractmethod
    def compute_gradients(self, hook_context: Context) -> List[torch.Tensor]:
        """
        The core computation method for a client.

        This method performs one step of local training: it samples a batch of data,
        computes the loss, and calculates the gradients. The key difference between
        HonestClient and ByzantineClient lies in the implementation of this method.

        Args:
            hook_context (HookContext): The context object providing shared state and
                                        information about the current training step.

        Returns:
            List[torch.Tensor]: A list of gradient tensors for each model parameter.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.client_id})>"
    
    def get_state(self) -> dict:
        """
        Returns the current state of the client for checkpointing.

        This includes the model parameters and any other relevant attributes.
        """
        state = {
            "model_state": self.model.state_dict() if self.model else None,
            "client_id": self.client_id,
            "client_type": getattr(self, "client_type", None),
            "state": self.other_state
        }
        return state

    def set_state(self, state: dict) -> None:
        """
        Loads the client state from a checkpoint.

        This restores the model parameters and other relevant attributes.
        """
        if self.model and state.get("model_state") is not None:
            self.model.load_state_dict(state["model_state"])
        # Restore other attributes if needed
        self.client_id = state.get("client_id", self.client_id)
        if "client_type" in state:
            self.client_type = state["client_type"]
        if "state" in state:
            self.other_state = state["state"]

class HonestClient(BaseClient):
    """
    An honest client that computes and returns gradients correctly.

    It follows the training protocol without any malicious modifications.
    """
    def __init__(
        self,
        client_id: int,
        data_partition: Optional[Subset] = None,
        sampler: Optional[BaseSampler] = None
    ):
        super().__init__(client_id, data_partition, sampler)
        self.client_type = "Honest"

    def compute_gradients(self, context: Context) -> List[torch.Tensor]:
        """
        Computes gradients based on a mini-batch of local data.

        This implementation is "honest" because it returns the gradients exactly
        as computed by the loss function's backward pass.
        """
        if not self.model or self.device is None:
            raise RuntimeError("Client model and device must be set before computing gradients.")

        self.model.train()
        self.model.zero_grad()

        # Get data from the sampler
        index, data, target = self.sampler.get_sample()
        index, data, target = index, data.to(self.device), target.to(self.device)
        context.index, context.data, context.target = index, data, target

        # Perform forward and backward passes
        output = self.model(data)
        loss = self.loss_fn(output, target)
        loss.backward()
        context.output = output
        context.loss = loss

        # Extract and return gradients
        gradients = [
            (p.grad.clone().detach() if p.grad is not None
             else torch.zeros_like(p))
            for p in self.model.parameters()
        ]
        return gradients


class ByzantineClient(BaseClient):
    """
    A Byzantine client that deliberately corrupts its computed gradients.

    This client applies a specified attack strategy to modify them before sending
    to the server. It performs standard forward/backward passes before applying
    malicious behavior, enabling realistic and modular attack design.

    Attributes:
        attack_strategy (BaseAttack): Defines how to corrupt the gradients.
    """
    def __init__(
        self,
        client_id: int,
        attack_strategy: BaseAttack,
        data_partition: Optional[Subset] = None,
        sampler: Optional[BaseSampler] = None
    ):
        super().__init__(client_id, data_partition, sampler)
        self.attack_strategy = attack_strategy
        self.client_type = "Byzantine"
        self.if_compute_grad = attack_strategy.if_byz_compute_grad

    def compute_gradients(self, context: Context) -> List[torch.Tensor]:
        """
        Performs standard local training step, then applies attack strategy to
        corrupt the computed gradients.
        """
        if not self.model or self.device is None:
            raise RuntimeError("Client model and device must be set before computing gradients.")

        context.current_client_id = self.client_id

        if self.if_compute_grad:
            self.model.train()
            self.model.zero_grad()

            index, data, target = self.sampler.get_sample()
            index, data, target = index, data.to(self.device), target.to(self.device)

            context.index = index
            context.data = data
            context.target = target

            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()

            context.output = output
            context.loss = loss

            message = [
                (p.grad.clone().detach() if p.grad is not None
                 else torch.zeros_like(p))
                for p in self.model.parameters()
            ]
            context.grad[self.client_id] = message
            context.gradients = message

        corrupted_gradients = self.attack_strategy.attack(context)

        return corrupted_gradients


    def __repr__(self) -> str:
        return f"<ByzantineClient(id={self.client_id}, attack={self.attack_strategy.__class__.__name__})>"
