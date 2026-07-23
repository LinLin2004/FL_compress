import torch
from typing import Dict
from fl_framework.core.hooks import hook_registry, HookType, Context
from fl_framework.components.optimizers.base_optimizer import BaseOptimizer


class LFRPG(BaseOptimizer):
    def __init__(self, lr: float, lambda_penalty=1.6,
                 delta0=0.003, delta_n=0.003,
                 mu=1e-3, step_interval=5,
                 L0=15, Ln=10, use_prox: bool = True):
        super().__init__(lr)
        self.lambda_penalty = lambda_penalty
        self.delta0 = delta0
        self.delta_n = delta_n
        self.mu = mu
        self.step_interval = step_interval
        self.L0 = L0
        self.Ln = Ln
        self.use_prox = use_prox
        self.beta_fn = lambda k: 2 / (k + 2)
        self.frame = 0
        self.wn = None
        self.un = None
        self.vn = None   
        self.w0 = None
        self.u0 = None
        self.v0 = None

        self.local_grad_acc = dict()

    def alpha_server_fn(self, k: int) -> float:
        return (self.delta0 / 14) * (k + 2) ** 2 + (3 / 2) * self.L0

    def alpha_client_fn(self, k: int) -> float:
        return (3 * self.delta_n / 14) * (k + 2) ** 2 + self.Ln

    def register_hooks(self):
        hook_registry.register(HookType.BEFORE_RUN, self._initialize)
        hook_registry.register(HookType.BEFORE_STEP_SERVER, self._maybe_broadcast_model)
        hook_registry.register(HookType.AFTER_COMPUTE, self._accumulate_local_gradient)
        hook_registry.register(HookType.BEFORE_AGGREGATE, self._average_local_gradient)

    @torch.no_grad()
    def _initialize(self, context: Context):
        server = context.server
        # self.w0 = [p.detach().clone() for p in server.model.parameters()]
        # self.v0 = [torch.zeros_like(p) for p in self.w0]
        # self.u0 = [torch.zeros_like(p) for p in self.w0]
        # self.wn = [[p.detach().clone() for p in client.model.parameters()] for client in server.clients]
        # self.un = [[torch.zeros_like(p) for p in client.model.parameters()] for client in server.clients]
        # self.vn = [[torch.zeros_like(p) for p in client.model.parameters()] for client in server.clients]
        self.w0 = [p.detach().clone() for p in server.model.parameters()]
        self.v0 = [p.detach().clone() for p in self.w0]
        self.u0 = [p.detach().clone() for p in self.w0]
        self.wn = [[p.detach().clone() for p in client.model.parameters()] for client in server.clients]
        self.un = [[p.detach().clone() for p in client.model.parameters()] for client in server.clients]
        self.vn = [[p.detach().clone() for p in client.model.parameters()] for client in server.clients]

    @torch.no_grad()
    def _maybe_broadcast_model(self, context: Context):
        if context.current_step % self.step_interval == 0:
            beta = self.beta_fn(self.frame)
            self.u0 = [(1 - beta) * w + beta * v for w, v in zip(self.w0, self.v0)]

            alpha = self.alpha_server_fn(self.frame)
            grads = [self._grad_f0(u) for u in self.u0]
            self.w0 = [u - (1 / alpha) * g for u, g in zip(self.u0, grads)]

            for p, w in zip(context.server.model.parameters(), self.w0):
                p.data.copy_(w)

            context.server.distribute_model()

    def _grad_f0(self, param):
        return self.delta0 * param

    def prox_huber_closed(self, z: torch.Tensor, alpha: float) -> torch.Tensor:
        norm = torch.norm(z)
        gamma = self.lambda_penalty / alpha
        threshold = self.mu + gamma
        if norm <= threshold:
            return (self.mu / (self.mu + gamma)) * z
        else:
            return (1 - gamma / norm) * z

    def _huber_grad(self, w0, wn):
        diff = w0 - wn
        norm = torch.norm(diff)
        return diff / self.mu if norm <= self.mu else diff / norm

    def _accumulate_local_gradient(self, context: Context):
        client_id = context.current_client_id
        client = context.clients[client_id]

        model = client.model
        alpha = self.alpha_client_fn(self.frame)
        beta = self.beta_fn(self.frame)
        g_n = []

        client_grad = context.grad[client_id]  # 从 context 取出的梯度 List[Tensor]
        un = self.un[client_id]
        vn = self.vn[client_id]
        wn = self.wn[client_id]

        # 依次处理每一层
        for u, w, v, p, p0, g in zip(un, vn, wn, model.parameters(), context.server.model.parameters(), client_grad):
            if self.use_prox:
                u.data.copy_((1-beta)*w+beta*v)
                z = p0.data - u + g / alpha
                prox = self.prox_huber_closed(z, alpha)
                w.data.copy_(p0.data - prox)
                p.data.copy_(w)
                huber_grad = self.lambda_penalty * self._huber_grad(p0, w)
                v.data.copy_(v - (self.delta_n * (v - u) + g - huber_grad) / (self.delta_n + alpha * beta))
            else:
                w.data.copy_(w.data - self.lr * g)
                p.data.copy_(w)
                huber_grad = self.lambda_penalty * self._huber_grad(p0.data, w)
            g_n.append(huber_grad)

        # z_list = []
        # for p0, u, g in zip(context.server.model.parameters(), self.u0, client_grad):
        #     z = p0.data - u + (1 / alpha) * g
        #     z_list.append(z)

        # for param, z, p0, g in zip(model.parameters(), z_list, context.server.model.parameters(), client_grad):
        #     if self.use_prox:
        #         new_val = self.prox_huber_closed(z, alpha)
        #         grad_g = self.lambda_penalty * self._huber_grad(p0.data, new_val)
        #         param.data.copy_(p0.data - new_val)
        #     else:
        #         new_val = param.data - self.lr * g
        #         param.data.copy_(new_val)
        #         grad_g = self.lambda_penalty * self._huber_grad(p0.data, new_val)
        #     g_n.append(grad_g)

        # 保存处理后的梯度
        self.local_grad_acc.setdefault(client_id, []).append(g_n)

    def _average_local_gradient(self, context: Context):
        for client_id, g_list in self.local_grad_acc.items():
            T = len(g_list)
            if T == 0:
                continue
            avg_grad = [sum(gs[i] for gs in g_list) / T for i in range(len(g_list[0]))]
            context.grad[client_id] = avg_grad
        self.local_grad_acc.clear()

    @torch.no_grad()
    def step(self, server, aggregated_grad):
        beta = self.beta_fn(self.frame)
        alpha = self.alpha_server_fn(self.frame)

        for i in range(len(self.v0)):
            grad_f0 = self._grad_f0(self.u0[i])
            self.v0[i] -= (self.delta0 * (self.v0[i] - self.u0[i]) + grad_f0 + aggregated_grad[i]) / (self.delta0 + alpha * beta)
        # for par in server.model.parameters():
        #     print(par)
        self.frame += 1

    def get_state(self) -> Dict:

        return {
            'w0': self.w0,
            'v0': self.v0,
            'u0': self.u0,
            'slot': self.frame
        }

    def set_state(self, state: Dict):
        self.w0 = state['w0']
        self.v0 = state['v0']
        self.u0 = state['u0']
        self.frame = state['slot']
