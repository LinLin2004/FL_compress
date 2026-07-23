# fl_framework/core/server.py

from __future__ import annotations
from typing import TYPE_CHECKING, List, Any, Callable
from copy import deepcopy

import torch
import torch.nn as nn
import logging


# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from fl_framework.core.client import BaseClient as Client
    from fl_framework.components.optimizers.base_optimizer import BaseOptimizer
    from fl_framework.components.aggregators.base_aggregator import BaseAggregator


class Server:
    """
    The central server in the federated learning setup.

    The Server is a "thin" component, acting primarily as a state container for the
    global model and a manager for the clients. It delegates complex tasks like
    gradient aggregation and model updates to other components (Aggregator, Optimizer)
    via the hook system.

    Attributes:
        model (nn.Module): The global model being trained.
        optimizer (BaseOptimizer): The optimizer responsible for applying updates.
                                   The server holds a reference but doesn't use it directly.
        aggregator (BaseAggregator): The component responsible for aggregating gradients.
        clients (List[Client]): The list of all clients registered with the server.
        device (torch.device): The device (e.g., 'cuda:0') where the model is located.
    """

    def __init__(
        self,
        model: nn.Module,
        aggregator: BaseAggregator,
        optimizer: BaseOptimizer,
        train_loss_fn: Callable = nn.CrossEntropyLoss(),
        test_loss_fn: Callable = nn.CrossEntropyLoss(),
        test_dataloader: Any = None,
        metrics: List[Callable] = [],
        devices: List[str] = ["cuda:0"],
    ):
        """
        Initializes the Server.

        Args:
            model (nn.Module): The initial global model.
            aggregator (BaseAggregator): The aggregation strategy to be used.
            optimizer (BaseOptimizer): The optimizer to be used.
            device (str): The device to run the server's model on.
        """
        self.devices = devices
        self.models_on_devices = {device: deepcopy(model).to(device) for device in self.devices}
        self.model = self.models_on_devices[self.devices[0]]  # Default model on the first device
        self.device = self.devices[0]
        self.train_loss_fn = train_loss_fn
        self.test_loss_fn = test_loss_fn
        self.metrics = metrics
        self.aggregator = aggregator
        self.optimizer: BaseOptimizer = optimizer  # Hold a reference for the Coordinator
        self.clients: List[Client] = []
        self.test_dataloader = test_dataloader
        self.other_state = {}

    def register_clients(self, clients: List[Client]):
        """
        Registers a list of clients with the server.

        Args:
            clients (List[Client]): The clients to be managed by the server.
        """
        self.clients = clients
        for i, client in enumerate(self.clients):
                assigned_device = self.devices[i % len(self.devices)]
                client.device = assigned_device
                client.model = deepcopy(self.models_on_devices[assigned_device])
                client.loss_fn = deepcopy(self.train_loss_fn)
        logging.info(f"[Server] Registered {len(self.clients)} clients.")


    @torch.no_grad()
    def distribute_model(self):
        """
        Distributes the current global model to all devices.
        This method ensures that the model is available on all specified devices.

        The client model is actually a reference to the server's model, 
        so all models on clinets are also updated.
        """
        state_dict = self.model.state_dict()

        for device in self.devices:
            self.models_on_devices[device].load_state_dict(state_dict)

        for client in self.clients:
            # if client.client_type != "Byzantine" and hasattr(client, "model"):
            client.model.load_state_dict(state_dict)

    def aggregate_gradients(self, gradients: List[Any]) -> Any:
        """
        Aggregates gradients received from clients by delegating to the aggregator.

        Args:
            gradients (List[Any]): A list of gradients from the clients.

        Returns:
            Any: The single aggregated gradient.
        """
        # The server itself doesn't know the aggregation logic. It just calls
        # the component responsible for it. This is a key design principle.
        return self.aggregator.aggregate(gradients)

    def test(self):
        """
        Runs a test pass on the server's model using the test dataloader.

        This method is typically called at the end of each round to evaluate
        the model's performance on unseen data.
        """
        if self.test_dataloader is None:
            print("[Server] No test dataloader provided. Skipping testing.")
            return

        self.model
        self.model.eval()
        avg_loss = []
        all_metrics = {metric.__name__: 0 for metric in self.metrics}
        with torch.no_grad():
            for batch in self.test_dataloader:
                # Process the batch
                _, inputs, targets = batch
                inputs, targets = inputs.to(self.devices[0]), targets.to(self.devices[0])
                outputs = self.model(inputs)
                loss = self.test_loss_fn(outputs, targets)
                avg_loss.append(loss.item())
                for metric in self.metrics:
                    all_metrics[metric.__name__] += metric(outputs, targets).item()
        self.model.train()

        # Average the metrics
        num_batches = len(self.test_dataloader)
        if num_batches > 0:
            avg_loss = sum(avg_loss) / num_batches
            for key in all_metrics:
                all_metrics[key] /= num_batches

        return avg_loss, all_metrics
    
    def get_state(self) -> dict:
        """
        Returns the current state of the server for checkpointing.

        This includes the global model parameters and any other relevant attributes.
        """
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.get_state(),
            "aggregator_state": self.aggregator.get_state() if hasattr(self.aggregator, "get_state") else None,
            "state": self.other_state,  # Additional state for custom attributes
        }
        return state

    def set_state(self, state: dict) -> None:
        """
        Loads the server state from a checkpoint.

        This restores the global model parameters and other relevant attributes.
        """
        if state.get("model_state") is not None:
            self.model.load_state_dict(state["model_state"])
        if "aggregator_state" in state and state["aggregator_state"] is not None and hasattr(self.aggregator, "set_state"):
            self.aggregator.set_state(state["aggregator_state"])
        if "optimizer_state" in state and state["optimizer_state"] is not None:
            self.optimizer.set_state(state["optimizer_state"])
        if "state" in state:
            self.other_state = state["state"]