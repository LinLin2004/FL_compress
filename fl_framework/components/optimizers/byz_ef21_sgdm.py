# fl_framework/components/optimizers/byz_ef21_sgdm.py
"""Byz-EF21-SGDM: Byzantine-robust EF21 with client-side SGDM and Top-k compression.

Combines:
  - Client-side stochastic gradient descent with momentum (SGDM)
  - EF21-style error feedback (compress the *change* Delta = v - g, not v itself)
  - Top-k sparsifier (keep the k largest-magnitude components)
  - Krum-based robust aggregation on the server (delegated to the aggregator)

Algorithm outline (per round t):
  Honest worker i:
    1. Compute stochastic gradient  s_i^(t) = grad l_i(x^(t); xi)
    2. Update local momentum         v_i^(t) = (1-eta)*v_i^(t-1) + eta*s_i^(t)
    3. Compute delta                 Delta_i^(t) = v_i^(t) - g_i^(t-1)
    4. Top-k compress                c_i^(t) = TopK(Delta_i^(t), k)
    5. Update local estimate         g_i^(t) = g_i^(t-1) + c_i^(t)
    6. Send c_i^(t) to server

  Byzantine worker j:
    - Sends an arbitrary forged c_j^(t) (the attack output)

  Server:
    7. Update global estimates       g_i^(t) = g_i^(t-1) + c_i^(t)   for all i
    8. Krum aggregate                g^(t) = Krum({g_i^(t)}, f)
    9. Update model                  x^(t+1) = x^(t) - gamma * g^(t)

The optimizer uses two hooks:
  - AFTER_COMPUTE: client-side momentum + EF21 error feedback + Top-k compression
                   (honest clients only; Byzantine clients are left untouched
                    so attacks can read all_honest_gradients and produce their
                    forged c_i^(t))
  - BEFORE_AGGREGATE: reconstruct per-client g_i^(t) on the server side
                      from the c_i^(t) stored in context.grad, using
                      g_i^(t) = g_i^(t-1) + c_i^(t) for ALL clients
                      (both honest and Byzantine).  The Byzantine attack
                      output is treated as a forged c_i^(t) and goes
                      through the same EF21 reconstruction, so the attack
                      effect accumulates through error feedback rather than
                      being injected directly.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch

from .base_optimizer import BaseOptimizer
from fl_framework.core.hooks import HookType, Context, hook_registry


class ByzEF21SGDM(BaseOptimizer):
    """Byzantine-robust EF21 with client-side SGDM and Top-k compression.

    Parameters
    ----------
    lr : float
        Global learning rate gamma.  Default 0.01.
    momentum : float
        Momentum coefficient eta (used as the weight for the new gradient).
        v = (1-eta)*v + eta*s, so eta=0.9 means heavy smoothing.  Default 0.9.
    compression_ratio : float
        Fraction of dimensions to keep in Top-k, i.e. k = ceil(ratio * d).
        Default 0.1 (keep 10% of components).
    """

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.9,
        compression_ratio: float = 0.1,
    ) -> None:
        super().__init__(lr=lr)
        self.momentum = momentum
        self.compression_ratio = compression_ratio

        # --- Lazy-initialised state (set on first gradient) ---
        self.d: Optional[int] = None           # total gradient dimension
        self.k: Optional[int] = None           # number of components to keep
        self.reference_shapes: Optional[List[torch.Size]] = None

        # Per-client momentum buffers: v_i^(t)
        # Indexed by client_id; each is a flat (d,)-shaped tensor.
        self._momentum_buffers: List[Optional[torch.Tensor]] = []

        # Per-client EF21 estimate buffers: g_i^(t-1)
        # These track the server-side estimate of each client's momentum.
        self._g_buffers: List[Optional[torch.Tensor]] = []

        # Server-side copies of g_i^(t-1), kept in sync with client-side.
        # After compression, the server reconstructs g_i^(t) = g_i^(t-1) + c_i^(t).
        self._server_g_buffers: List[Optional[torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def register_hooks(self) -> None:
        super().register_hooks()
        hook_registry.register(HookType.AFTER_COMPUTE, self._client_process)
        hook_registry.register(HookType.BEFORE_AGGREGATE, self._server_reconstruct)

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _lazy_init(self, sample_grad: List[torch.Tensor], device: torch.device) -> None:
        """Initialise dimensions and buffers on the first gradient computation."""
        self.reference_shapes = [g.shape for g in sample_grad]
        flat = self._flatten(sample_grad)
        self.d = flat.numel()
        self.k = max(1, math.ceil(self.compression_ratio * self.d))

        print(
            f"[Byz-EF21-SGDM] Initialised: d={self.d}, k={self.k}, "
            f"compression_ratio={self.compression_ratio}"
        )
        print(
            f"[Byz-EF21-SGDM] Top-k keeps {self.k}/{self.d} components "
            f"({self.k / self.d * 100:.1f}%)"
        )

    # ------------------------------------------------------------------
    # Gradient flatten / unflatten
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(grad_list: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate a list of parameter-shaped tensors into a flat vector."""
        return torch.cat([g.reshape(-1) for g in grad_list])

    def _unflatten(self, flat: torch.Tensor) -> List[torch.Tensor]:
        """Reshape a flat (d,)-vector back to the original parameter shapes."""
        result: List[torch.Tensor] = []
        offset = 0
        for shape in self.reference_shapes:
            n = shape.numel()
            result.append(flat[offset: offset + n].view(shape))
            offset += n
        return result

    # ------------------------------------------------------------------
    # Top-k compression
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _topk(vec: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k sparsification: keep the k components with largest absolute
        value, set the rest to zero.

        Returns a sparse vector of the same shape as *vec*.
        """
        if k >= vec.numel():
            return vec.clone()
        # Find the k largest absolute values
        _, topk_indices = torch.topk(vec.abs(), k)
        result = torch.zeros_like(vec)
        result.scatter_(0, topk_indices, vec[topk_indices])
        return result

    # ------------------------------------------------------------------
    # Hook 1 — AFTER_COMPUTE: client-side momentum + EF21 + Top-k
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _client_process(self, context: Context) -> None:
        """Per-client processing: momentum update, EF21 error feedback, Top-k.

        Honest clients get the full pipeline.  Byzantine clients are skipped
        so that their attack output remains in *context.grad* as a per-layer
        gradient list; this is required because later Byzantine clients may
        read *context.all_honest_gradients* during their attack.  The attack
        output will be treated as a forged c_i^(t) in the server-side
        reconstruction step.
        """
        client_id = context.current_client_id
        client = context.clients[client_id]
        client_grad = context.grad[client_id]

        # --- Lazy init on first call ---
        if self.d is None:
            self._lazy_init(client_grad, client_grad[0].device)

        # --- Ensure buffer lists are large enough ---
        while len(self._momentum_buffers) <= client_id:
            self._momentum_buffers.append(None)
        while len(self._g_buffers) <= client_id:
            self._g_buffers.append(None)

        # --- Byzantine clients: leave gradient untouched ---
        if client.client_type == "Byzantine":
            return

        # --- Honest client processing ---
        device = client_grad[0].device
        flat_grad = self._flatten(client_grad).to(device)  # s_i^(t)

        # 1. Update local momentum: v_i^(t) = (1-eta)*v_i^(t-1) + eta*s_i^(t)
        v_buf = self._momentum_buffers[client_id]
        if v_buf is None:
            # First iteration: v_i^(0) = s_i^(0)
            v_buf = flat_grad.clone()
        else:
            v_buf = v_buf.to(device)
            v_buf.mul_(1.0 - self.momentum).add_(flat_grad, alpha=self.momentum)
        self._momentum_buffers[client_id] = v_buf

        # 2. Compute delta: Delta_i^(t) = v_i^(t) - g_i^(t-1)
        g_buf = self._g_buffers[client_id]
        if g_buf is None:
            # g_i^(-1) = 0, so Delta = v
            delta = v_buf.clone()
        else:
            g_buf = g_buf.to(device)
            delta = v_buf - g_buf

        # 3. Top-k compress: c_i^(t) = TopK(Delta_i^(t), k)
        c = self._topk(delta, self.k)

        # 4. Update local estimate: g_i^(t) = g_i^(t-1) + c_i^(t)
        if g_buf is None:
            g_buf = c.clone()
        else:
            g_buf = g_buf.to(device)
            g_buf.add_(c)
        self._g_buffers[client_id] = g_buf

        # 5. Replace context.grad with the compressed vector c_i^(t)
        #    The server will reconstruct g_i^(t) from c_i^(t) + g_i^(t-1).
        #    We store c_i^(t) as a singleton list to match the
        #    List[List[Tensor]] interface expected by aggregators.
        context.grad[client_id] = [c]

    # ------------------------------------------------------------------
    # Hook 2 — BEFORE_AGGREGATE: server-side reconstruction of g_i^(t)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _server_reconstruct(self, context: Context) -> None:
        """Reconstruct each client's g_i^(t) on the server side.

        For ALL clients (both honest and Byzantine), context.grad[i] contains
        the delta c_i^(t).  The server maintains g_i^(t-1) and reconstructs:
            g_i^(t) = g_i^(t-1) + c_i^(t)

        - Honest clients: c_i^(t) is the Top-k compressed delta, stored as
          a singleton list [flat_tensor] with shape (d,).
        - Byzantine clients: c_i^(t) is the attack output, stored as a
          per-layer gradient list.  We flatten it to (d,) and treat it as
          the forged c_i^(t), then apply the same reconstruction formula.
          This means the Byzantine attack is accumulated through the EF21
          error feedback mechanism rather than injected directly.

        After reconstruction, each context.grad[i] is replaced with the
        full g_i^(t) vector (as a singleton list) so that the Krum
        aggregator can operate on the reconstructed estimates.
        """
        for i in range(len(context.grad)):
            g = context.grad[i]
            if g is None:
                continue

            # Ensure server-side buffer list is large enough
            while len(self._server_g_buffers) <= i:
                self._server_g_buffers.append(None)

            # Determine c_i^(t) from context.grad[i]
            # Honest clients: [flat (d,)-tensor] (singleton list from _client_process)
            # Byzantine clients: per-layer list (attack output from ByzantineClient)
            if len(g) == 1 and g[0].dim() == 1 and g[0].numel() == self.d:
                # Honest client: c_i^(t) is already a flat (d,)-tensor
                c_i = g[0]
            else:
                # Byzantine client: flatten the attack output to get forged c_i^(t)
                c_i = self._flatten(g)

            # Reconstruct: g_i^(t) = g_i^(t-1) + c_i^(t)
            device = c_i.device
            server_g = self._server_g_buffers[i]
            if server_g is None:
                # g_i^(-1) = 0, so g_i^(t) = c_i^(t)
                server_g = c_i.clone()
            else:
                server_g = server_g.to(device)
                server_g.add_(c_i)
            self._server_g_buffers[i] = server_g

            # Replace with the reconstructed g_i^(t)
            context.grad[i] = [server_g]

    # ------------------------------------------------------------------
    # Server-side update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, server, aggregated_grad) -> None:
        """Apply the Krum-aggregated gradient to update the global model.

        Parameters
        ----------
        server : Server
            The central server holding the global model.
        aggregated_grad : List[Tensor]
            A singleton list containing the (d,)-shaped flat vector
            selected by Krum (the winning client's g_i^(t)).
        """
        if server is None or aggregated_grad is None:
            print(
                "[Byz-EF21-SGDM] Warning: server or aggregated_grad is None, "
                "skipping update."
            )
            return

        # Krum returns [flat_vector] — the winning client's g_i^(t).
        g_aggregated = aggregated_grad[0]  # shape (d,)

        device = next(server.model.parameters()).device
        g_aggregated = g_aggregated.to(device, non_blocking=True)

        # Reshape to per-parameter tensors and apply update: x^(t+1) = x^(t) - gamma * g^(t)
        updates = self._unflatten(g_aggregated)

        for param, update in zip(server.model.parameters(), updates):
            param.data.add_(update.to(param.device), alpha=-self.lr)

    # ------------------------------------------------------------------
    # State persistence (checkpoint / resume)
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return serialisable state for checkpointing."""
        state: dict = {
            "lr": self.lr,
            "momentum": self.momentum,
            "compression_ratio": self.compression_ratio,
            "d": self.d,
            "k": self.k,
            "reference_shapes": self.reference_shapes,
        }

        # Momentum buffers — move to CPU for serialisation.
        if self._momentum_buffers:
            state["momentum_buffers"] = [
                buf.cpu().clone() if buf is not None else None
                for buf in self._momentum_buffers
            ]

        # Client-side g buffers
        if self._g_buffers:
            state["g_buffers"] = [
                buf.cpu().clone() if buf is not None else None
                for buf in self._g_buffers
            ]

        # Server-side g buffers
        if self._server_g_buffers:
            state["server_g_buffers"] = [
                buf.cpu().clone() if buf is not None else None
                for buf in self._server_g_buffers
            ]

        return state

    def set_state(self, state: dict) -> None:
        """Restore state from a checkpoint."""
        self.lr = state.get("lr", self.lr)
        self.momentum = state.get("momentum", self.momentum)
        self.compression_ratio = state.get("compression_ratio", self.compression_ratio)
        self.d = state.get("d")
        self.k = state.get("k")
        self.reference_shapes = state.get("reference_shapes")

        raw_mom = state.get("momentum_buffers")
        if raw_mom is not None:
            self._momentum_buffers = [
                buf.clone() if buf is not None else None for buf in raw_mom
            ]

        raw_g = state.get("g_buffers")
        if raw_g is not None:
            self._g_buffers = [
                buf.clone() if buf is not None else None for buf in raw_g
            ]

        raw_sg = state.get("server_g_buffers")
        if raw_sg is not None:
            self._server_g_buffers = [
                buf.clone() if buf is not None else None for buf in raw_sg
            ]
