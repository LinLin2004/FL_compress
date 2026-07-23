# fl_framework/core/coordinator.py

from __future__ import annotations
from typing import TYPE_CHECKING, List, Any
import time
import glob
import os
import logging
from tqdm import tqdm
# from fl_framework.components.optimizers.param_avg_optimizer import ParamAvgOptimizer

import torch
from fl_framework.core.hooks import hook_registry, HookType, Context

# For type hinting only
if TYPE_CHECKING:
    from fl_framework.core.server import Server
    from fl_framework.core.client import BaseClient as Client

def client_trigger(clients, context, hook_type: HookType):
    """Hook trigger"""
    for c in clients:
        context.current_client_id = c.client_id
        hook_registry.trigger(hook_type, context)

def client_task(clients, context) -> Any:
    """ This function prepares the task for each client. """
    # Create a per-client context for thread-safety in a real parallel scenario
    for c in clients:
        context.current_client_id = c.client_id

        hook_registry.trigger(HookType.BEFORE_COMPUTE, context)
        
        raw_grad = c.compute_gradients(context)
        # print(raw_grad)
        context.grad[c.client_id] = raw_grad
        
        hook_registry.trigger(HookType.AFTER_COMPUTE, context)
        # The gradient to return may have been modified by the AFTER_COMPUTE hook


class Coordinator:
    """
    The process driver for the federated learning experiment.

    The Coordinator orchestrates the entire training process by following a strict
    sequence of operations: client selection, model distribution, local computation,
    gradient aggregation, and global model update.

    It is the central component that triggers all hooks at the appropriate times.
    It remains completely agnostic to the specific optimization algorithm,
    delegating all complex logic to the Optimizer and other components via the

    hook system. This adherence to the "Optimizer-Centric" design is paramount.
    """

    def __init__(
        self,
        server: Server,
        clients: List[Client],
        num_byzantine: int,
        # resource_manager: BaseResourceManager,
        # Configuration for the experiment run
        num_rounds: int,
        num_round_steps: int,
        save_path: str = None,
    ):
        """
        Initializes the Coordinator.

        Args:
            server (Server): The central server instance.
            clients (List[Client]): A list of all client instances.
            resource_manager (ResourceManager): The manager for parallel execution.
            num_rounds (int): Total number of communication rounds.
            num_local_steps (int): Number of local training steps per client per round.
        """
        self.server = server
        self.clients = clients
        self.optimizer = server.optimizer # Get optimizer from server
        # self.resource_manager = resource_manager
        self.num_byzantine = num_byzantine  # Number of Byzantine clients

        # Experiment parameters:
        self.save_path = save_path
        self.num_rounds = num_rounds
        self.num_local_steps = num_round_steps
        self.history = {'loss':[], 'metrics':[]}

        # --- Critical Setup Steps ---
        # Register clients with the server
        self.server.register_clients(self.clients)

        # Initialize the master HookContext
        self.context = Context()
        self.context.server = self.server
        self.context.clients = self.clients
        self.context.optimizer = self.server.optimizer
        self.context.aggregator = self.server.aggregator
        self.context.grad = [None for _ in self.clients] # Placeholder for raw gradients

        self.optimizer.register_hooks()

    def run(self, resume_from: str = None) -> None:
        """
        Starts and executes the entire federated learning experiment.
        """
        logging.info("="*30)
        logging.info("Federated Learning Experiment Starting")
        logging.info(f"  - Rounds: {self.num_rounds}")
        logging.info(f"  - Local Steps: {self.num_local_steps}")
        logging.info("="*30)

        start_time = time.time()

        if resume_from:
            self.resume(resume_from)
            start_round = self.context.current_round
        else:
            start_round = 0
            hook_registry.trigger(HookType.BEFORE_RUN, self.context)

        loss, metrics = self.server.test()
        self.history['loss'].append(loss)
        self.history['metrics'].append(metrics)
        logging.info(f"[Coordinator] Round {0} Loss: {loss}")
        for k, v in metrics.items():
            logging.info(f"[Coordinator] Round {0} Metric {k}: {v}")

        for rr in range(start_round, self.num_rounds):
            self.context.current_round = rr
            self._run_round()
            # Remove previous round checkpoints, keep only the latest
            for old_ckpt in glob.glob(os.path.join(self.save_path, "state_round_*.pth")):
                os.remove(old_ckpt)
            self.save_state(os.path.join(self.save_path, f"state_round_{rr}.pth"))

        hook_registry.trigger(HookType.AFTER_RUN, self.context)
        # self.resource_manager.shutdown()

        end_time = time.time()
        logging.info("\nFederated Learning Experiment Finished.")
        logging.info(f"Total duration: {end_time - start_time:.2f} seconds.")


    def _run_round(self) -> None:
        """Executes a single round."""
        round_num = self.context.current_round
        logging.info(f"\n--- Round {round_num + 1}/{self.num_rounds} ---")

        # Trigger Pre-Round Hooks
        hook_registry.trigger(HookType.BEFORE_ROUND_SERVER, self.context)
        client_trigger(self.clients, self.context, HookType.BEFORE_ROUND_CLIENT)
            
        # Training Steps
        for step in tqdm(list(range(self.num_local_steps))):
            self.context.current_step = step
            self._run_step()

        # Trigger Post-Round Hooks
        hook_registry.trigger(HookType.AFTER_ROUND_SERVER, self.context)
        client_trigger(self.clients, self.context, HookType.POST_ROUND_CLIENT)

        # Test
        loss, metrics = self.server.test()
        self.history['loss'].append(loss)
        self.history['metrics'].append(metrics)
        logging.info(f"[Coordinator] Round {round_num + 1} Loss: {loss}")
        for k, v in metrics.items():
            logging.info(f"[Coordinator] Round {round_num + 1} Metric {k}: {v}")


    def _run_step(self) -> List[Any]:
        """Executes one step of client computation and returns aggregated gradients."""
        # Reset gradients from previous step to avoid stale data
        self.context.grad = [None for _ in self.clients]

        hook_registry.trigger(HookType.BEFORE_STEP_SERVER, self.context)
        client_trigger(self.clients, self.context, HookType.BEFORE_STEP_CLIENT)

        if self.num_byzantine > 0:
            client_task(self.clients[:-self.num_byzantine], self.context)
            self.context.all_honest_gradients = self.context.grad[:-self.num_byzantine]
            client_task(self.clients[-self.num_byzantine:], self.context)
        else:
            client_task(self.clients, self.context)
            self.context.all_honest_gradients = self.context.grad

        all_gradients = self.context.grad

        step_interval = getattr(self.optimizer, "step_interval", 1)
        if self.context.current_step % step_interval == 0:
            # Only aggregate and update every step_interval steps
            hook_registry.trigger(HookType.BEFORE_AGGREGATE, self.context)
            aggregated_grad = self.server.aggregate_gradients(all_gradients)
            self.context.aggregated_grad = aggregated_grad
            hook_registry.trigger(HookType.AFTER_AGGREGATE, self.context)
            
            hook_registry.trigger(HookType.BEFORE_UPDATE, self.context)
            self.optimizer.step(self.server, aggregated_grad)
            hook_registry.trigger(HookType.AFTER_UPDATE, self.context)

            self.server.distribute_model()

            # for sp, cp in zip(self.server.model.parameters(), self.clients[0].model.parameters()):
            #     print((sp == cp).all())

            # 清理状态
            self.context.aggregated_grad = None

        hook_registry.trigger(HookType.AFTER_STEP_SERVER, self.context)
        client_trigger(self.clients, self.context, HookType.AFTER_STEP_CLIENT)

        self.context.all_honest_gradients = None

    def save_state(self, filepath: str) -> None:
        """
        Save the current experiment state to a file for resuming later.
        """
        state = {
            "current_round": self.context.current_round,
            "history": self.history,
            "server_state": self.server.get_state(),
        }
        torch.save(state, filepath)
        logging.info(f"[Coordinator] State saved to {filepath}")

    def resume(self, filepath: str) -> None:
        """
        Resume the experiment from a saved state file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"State file {filepath} not found.")
        state = torch.load(filepath)
        self.context.current_round = state["current_round"]
        self.server.set_state(state["server_state"])
        self.history = state['history']
        logging.info(f"[Coordinator] State loaded from {filepath}")
