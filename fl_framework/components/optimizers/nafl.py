from __future__ import annotations

from typing import List
import torch

from .base_optimizer import BaseOptimizer
from fl_framework.core.hooks import HookType, Context, hook_registry


class NALF(BaseOptimizer):
    """Server‑side **SGD** with双辅助序列 (zₖ, yₖ)。
    .. math::
        z_{k+1} &= \beta z_k + \nabla_k \\
        y_k     &= \beta z_{k+1} + \nabla_k \\
        x_{k+1} &= x_k - \eta y_k

    其中
      * :math:`x_k` — 全局模型参数（`server.model.parameters()`）
      * :math:`\nabla_k` — 已聚合的全局梯度 `aggregated_grad`
      * :math:`\beta` — `momentum`
      * :math:`\eta` — `lr`
    """

    def __init__(self, lr: float, weight_decay: float = 0.00005, momentum: float = 0.0) -> None:
        super().__init__(lr=lr)
        self.weight_decay = weight_decay
        self.momentum = momentum

        # 缓冲序列：与参数同形状，惰性初始化
        self._z: List[torch.Tensor] | None = None
        self._y: List[torch.Tensor] | None = None


    def register_hooks(self):
        super().register_hooks()
        hook_registry.register(HookType.AFTER_COMPUTE, self._weight_decay)

    @torch.no_grad()
    def _weight_decay(self, context: Context):
        client_id = context.current_client_id
        client = context.clients[client_id]
        if client.client_type != 'Byzantine':
            for grad, param in zip(context.grad[client_id], client.model.parameters()):
                if grad is not None:
                    grad.data.add_(param.data, alpha=self.weight_decay)


    @torch.no_grad()
    def step(self, server, aggregated_grad) -> None:
        """根据 (z, y) 序列公式更新 **server.model**。

        Args:
            server:           持有全局模型的服务器实例。
            aggregated_grad:  已在客户端层面聚合好的梯度列表 ∇ₖ。
        """
        if server is None or aggregated_grad is None:
            print("[SGD] Warning: server or aggregated_grad is None – skipping update.")
            return

        model = server.model

        # 惰性初始化 z、y 缓冲为零；保证设备与形状匹配
        if self._z is None:
            self._z = [torch.zeros_like(p, device=p.device) for p in model.parameters()]
            self._y = [torch.zeros_like(p, device=p.device) for p in model.parameters()]

        beta = self.momentum
        # print(next(model.parameters()).grad.sum())
        # print(aggregated_grad[0].sum())

        for i, (param, grad) in enumerate(zip(model.parameters(), aggregated_grad)):
            g = grad.to(param.device)

            # z_{k+1} = β z_k + g
            self._z[i].data.mul_(beta).add_(g)

            # y_k = β z_{k+1} + g
            self._y[i].data.mul_(0).add_(self._z[i]).mul_(beta).add_(g)
            # 上面两行等价于: self._y[i] = beta * self._z[i] + g

            # x_{k+1} = x_k - η y_k
            param.data.add_(self._y[i], alpha=-self.lr)
            # param.data.add_(g * self.lr)

    def get_state(self) -> dict:
        state: dict = {"lr": self.lr, "momentum": self.momentum}
        if self._z is not None:
            state["z"] = [z.cpu() for z in self._z]
            state["y"] = [y.cpu() for y in self._y]
        return state

    def set_state(self, state: dict) -> None:
        self.lr = state.get("lr", self.lr)
        self.momentum = state.get("momentum", self.momentum)
        z_list = state.get("z")
        y_list = state.get("y")
        if z_list is not None and y_list is not None:
            self._z = [z.clone() for z in z_list]
            self._y = [y.clone() for y in y_list]
