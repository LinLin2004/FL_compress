# fl_framework/components/attacks/base_attack.py

from abc import ABC, abstractmethod
from typing import List
import torch
from fl_framework.core.hooks import Context

class BaseAttack(ABC):
    """
    Abstract base class for all Byzantine attack strategies.

    An attack strategy is a modular component responsible for taking a list
    of honestly computed gradients and returning a corrupted version.
    This design keeps the client logic simple and decouples it from the
    specifics of any given attack.
    """

    if_byz_compute_grad = False

    @torch.no_grad()
    @abstractmethod
    def attack(self, context: Context) -> List[torch.Tensor]:
        """
        Applies the Byzantine attack.

        Args:
            gradients (List[torch.Tensor]): The list of honestly computed
                                    all gradient tensors from all honest clients.

        Returns:
            List[torch.Tensor]: The list of corrupted gradient tensors.
        """
        raise NotImplementedError

