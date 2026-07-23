# fl_framework/core/hooks.py

from __future__ import annotations
import collections
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from multiprocessing import Manager

# Use TYPE_CHECKING to avoid circular imports at runtime
# These are only for type hinting purposes
if TYPE_CHECKING:
    import torch
    from fl_framework.core.coordinator import Coordinator
    from fl_framework.core.server import Server
    from fl_framework.core.client import BaseClient as Client
    from fl_framework.components.optimizers.base_optimizer import BaseOptimizer
    from fl_framework.components.aggregators.base_aggregator import BaseAggregator


class HookType(Enum):
    """
    Enumeration of all available hook points in the training lifecycle.
    
    This ensures type-safety and prevents typos when registering hooks.
    """
    # Run-level hooks
    BEFORE_RUN = auto()
    AFTER_RUN = auto()

    # Round-level hooks
    BEFORE_ROUND_SERVER = auto()
    AFTER_ROUND_SERVER = auto()
    BEFORE_ROUND_CLIENT = auto()
    POST_ROUND_CLIENT = auto()

    # Step-level hooks (within a round, for each client's computation)
    BEFORE_STEP_CLIENT = auto()
    AFTER_STEP_CLIENT = auto()
    BEFORE_STEP_SERVER = auto()
    AFTER_STEP_SERVER = auto()

    # Granular operation hooks
    BEFORE_COMPUTE = auto()
    AFTER_COMPUTE = auto()
    BEFORE_AGGREGATE = auto()
    AFTER_AGGREGATE = auto()
    BEFORE_UPDATE = auto()
    AFTER_UPDATE = auto()


@dataclass
class Context:
    """
    A data container for passing state between components via hooks.
    
    This object is created by the Coordinator and passed to every triggered hook.
    Hooks can read from and write to this context to communicate and modify state.
    Attributes are optional as not all hooks will have access to all pieces of state.
    """
    # Core Components
    server: Optional[Server] = None
    clients: List[Client] = None # all clients
    optimizer: Optional[BaseOptimizer] = None
    aggregator: Optional[BaseAggregator] = None
    current_client_id: Optional[int] = None

    index: Optional[Any] = None # index of current data
    data: Optional[Any] = None  # Any data that needs to be passed around
    target: Optional[Any] = None  # The target value for supervised learning
    output: Optional[Any] = None  # Model output from the last forward pass
    loss: Optional[Any] = None


    grad: List[Any] = None          # Raw gradient computed by clients
    all_honest_gradients: List[Any] = None # Gradients from honest clients
    aggregated_grad: Any = None   # Gradient after being aggregated by the server

    # Loop counters
    current_round: int = -1
    current_step: int = -1
    

    # Extra storage for custom data
    extra: Dict[str, Any] = field(default_factory=dict)


class HookRegistry:
    """
    A central registry for managing and triggering hooks.
    
    This class follows a singleton-like pattern by providing a single global instance.
    Components (like Optimizers) register their callback functions here, and the
    Coordinator triggers them at the appropriate points in the training loop.
    """

    def __init__(self):
        """Initializes the hook registry."""
        self._hooks: Dict[HookType, List[Callable[[Context], None]]] = \
            collections.defaultdict(list)

    def register(self, hook_type: HookType, func: Callable[[Context], None]):
        """
        Registers a callback function for a specific hook type.

        Args:
            hook_type (HookType): The hook event to register for.
            func (Callable[[HookContext], None]): The function to be called when the
                                                 hook is triggered. It must accept a
                                                 HookContext object as its only argument.
        """
        self._hooks[hook_type].append(func)
        # print(f"[HookRegistry] Registered function '{func.__qualname__}' for hook '{hook_type.name}'")

    def trigger(self, hook_type: HookType, context: Context):
        """
        Triggers all registered callbacks for a specific hook type.

        Args:
            hook_type (HookType): The hook event to trigger.
            context (HookContext): The context object to pass to the callbacks.
        """
        # print(f"[HookRegistry] Triggering hook '{hook_type.name}'...")
        for func in self._hooks[hook_type]:
            func(context)

# All parts of the framework will import and use this single instance.
hook_registry = HookRegistry()
