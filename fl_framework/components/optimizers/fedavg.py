import torch
from fl_framework.components.optimizers.base_optimizer import BaseOptimizer
from fl_framework.core.hooks import hook_registry, HookType, Context
from typing import List


class FedAvgOptimizer(BaseOptimizer):
    """
    简化版参数平均优化器：只负责使用外部提供的参数更新 server 模型。
    不负责聚合、不更新客户端模型。
    """
    def __init__(self, lr: float, weight_decay: float = 0.001,step_interval: int = 10):
        super().__init__(lr)
        self.weight_decay = weight_decay
        self.step_interval = step_interval

    def register_hooks(self):
        """
        注册 hook：在客户端训练完成后缓存其模型参数。
        """
        hook_registry.register(HookType.AFTER_COMPUTE, self._cache_client_model_params)
    
    @torch.no_grad()
    def _cache_client_model_params(self, context: Context):
        client_id = context.current_client_id
        client = context.clients[client_id]

        if client.client_type == 'Byzantine' or client.model is None:
            return

        model = client.model
        param_list = []

        for param in model.parameters():
            if param.grad is None:
                raise ValueError(f"[Optimizer] Param {param.shape} has no grad.")

            w = param.detach().cpu().clone()
            g = param.grad.detach().cpu().clone()

            updated = w - self.lr * (g + self.weight_decay * w)
            param_list.append(updated)
            param.data.copy_(updated.to(param.device))

        if context.grad is None or len(context.grad) != len(context.clients):
            context.grad = [None] * len(context.clients)

        context.grad[client_id] = param_list


    @torch.no_grad()
    def step(self, server, aggregated_grad):
        if not isinstance(aggregated_grad, list):
            raise ValueError("Expected aggregated_grad to be List[Tensor].")

        # server_state = server.model.state_dict()
        # new_state = {
        #     k: v.to(server_state[k].device).type(server_state[k].dtype)
        #     for k, v in zip(server_state.keys(), aggregated_grad)
        # }
        # server.model.load_state_dict(new_state)

        model = server.model
        for param, updated in zip(model.parameters(), aggregated_grad):
            param.data.copy_(updated.to(param.device))

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict):
        pass
