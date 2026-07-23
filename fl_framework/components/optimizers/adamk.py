from __future__ import annotations

from typing import List
import torch

from .base_optimizer import BaseOptimizer
from fl_framework.core.hooks import HookType, Context, hook_registry


class ADAMK(BaseOptimizer):
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

    def __init__(self, lr: float, weight_decay: float = 0.00005, beta1: float = 0.9, beta2: float = 0.999, eps=1e-08, lr_decay_rate = 1., lr_decay_milestone=[5, 7]) -> None:
        super().__init__(lr=lr)
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_milestone = lr_decay_milestone
        self.lr_decay_milestone = [0] + self.lr_decay_milestone
        self.base_lr = lr

        # 缓冲序列：与参数同形状，惰性初始化
        self._v: List[torch.Tensor] | None = None
        self._m: List[torch.Tensor] | None = None


    def register_hooks(self):
        super().register_hooks()
        # hook_registry.register(HookType.BEFORE_RUN, self.init_state)
        hook_registry.register(HookType.AFTER_COMPUTE, self._weight_decay)
        hook_registry.register(HookType.BEFORE_UPDATE, self.get_lr)
    
    # def init_state(self, context: Context):
    #     server = context.server
    #     model = server.model
    #     self._v = [torch.zeros_like(p, device=p.device) for p in model.parameters()]
    #     self._m = [torch.zeros_like(p, device=p.device) for p in model.parameters()]
    
    def get_lr(self, context):
        round = context.current_round + 1
        for i in range(len(self.lr_decay_milestone) - 1, -1, -1):
            if round >= self.lr_decay_milestone[i]:
                break
        # print(i)
        self.lr = self.base_lr * self.lr_decay_rate ** i
        # print(f"round {round} lr {self.lr}")

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
        """根据 (z, y) 序列公式更新 **server.model**。

        Args:
            server:           持有全局模型的服务器实例。
            aggregated_grad:  已在客户端层面聚合好的梯度列表 ∇ₖ。
        """
        if server is None or aggregated_grad is None:
            print("[SGD] Warning: server or aggregated_grad is None - skipping update.")
            return

        
        model = server.model

        if self._v is None:
            self._m = [g.clone().detach() for g in aggregated_grad]
            self._v = [(g*g).clone().detach() for g in aggregated_grad]
        else:
            torch._foreach_mul_(self._m, self.beta1)
            torch._foreach_add_(self._m, aggregated_grad, alpha=(1 - self.beta1))

            torch._foreach_mul_(self._v, self.beta2)
            torch._foreach_addcmul_(self._v, aggregated_grad, aggregated_grad, value=(1 - self.beta2))

        params = list(model.parameters())
        denom = [v.clone() for v in self._v]
        torch._foreach_sqrt_(denom)
        torch._foreach_add_(denom, self.eps)
        torch._foreach_addcdiv_(params, self._m, denom, value=-self.lr)

    def get_state(self) -> dict:
        state: dict = {"lr": self.lr, 'base_lr': self.base_lr, "beta1": self.beta1, "beta2": self.beta2, "eps": self.eps}
        if self._v is not None:
            state["v"] = [z.cpu() for z in self._v]
            state["m"] = [m.cpu() for m in self._m]
        return state

    def set_state(self, state: dict) -> None:
        self.lr = state.get("lr", self.lr)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2 = state.get("beta2", self.beta2)
        v_list = state.get("v")
        m_list = state.get('m')
        self.base_lr = state.get("base_lr", self.base_lr)
        # y_list = state.get("y")
        if v_list is not None:
            self._v = [v.clone() for v in v_list]
            self._m = [m.clone() for m in m_list]
            # self._y = [y.clone() for y in y_list]
