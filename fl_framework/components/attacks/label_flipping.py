from typing import List
import torch
from .base_attack import BaseAttack, Context

class LabelFlipping(BaseAttack):
    """
    Implements the Label flipping attack.
    """

    if_byz_compute_grad = False

    def attack(self, context: Context) -> List[torch.Tensor]:

        clirent_id = context.current_client_id
        client = context.clients[clirent_id]

        client.model.train()
        client.model.zero_grad()

        index, data, target = client.sampler.get_sample()
        index, data, target = index, data.to(client.device), target.to(client.device)

        num_classes = client.data_partition.dataset.num_classes
        flipped_target = (num_classes - 1) - target

        context.index = index
        context.data = data
        context.target = target

        output = client.model(data)

        loss = client.loss_fn(output, flipped_target)
        loss.backward()

        context.output = output
        context.loss = loss

        return [p.grad.clone().detach() for p in client.model.parameters() if p.grad is not None]
