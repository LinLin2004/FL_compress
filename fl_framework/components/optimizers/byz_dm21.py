# fl_framework/components/optimizers/byz_dm21.py
"""Byz-DM21 / Byz-VR-DM21: Byzantine-robust distributed momentum with Top-k compression.

Combines:
  - Client-side dual momentum (v_i raw momentum, u_i auxiliary momentum = EMA of v_i)
  - Optional variance reduction (Byz-VR-DM21) using the SARAH/SPIDER trick
  - EF21-style error feedback (compress delta = u_i - g_i, not u_i itself)
  - Top-k sparsifier (keep the k largest-magnitude components)
  - Krum-based robust aggregation on the server (delegated to the aggregator)

Algorithm outline (per round t):

  Honest worker i:
    1. Compute stochastic gradient  s_i^(t) = grad l_i(x^(t); xi)
    2a. (Byz-DM21)  Update raw momentum:  v_i^(t) = (1-eta)*v_i^(t-1) + eta*s_i^(t)
    2b. (Byz-VR-DM21) Update raw momentum:
            v_i^(t) = s_i^(t) + (1-eta)*(v_i^(t-1) - s_i^(t-1))
        where s_i^(t-1) is the gradient at the previous model x^(t-1)
    3. Update auxiliary momentum:  u_i^(t) = (1-eta)*u_i^(t-1) + eta*v_i^(t)
    4. Compute delta:  Delta_i^(t) = u_i^(t) - g_i^(t-1)
    5. Top-k compress:  c_i^(t) = TopK(Delta_i^(t), k)
    6. Update local estimate:  g_i^(t) = g_i^(t-1) + c_i^(t)
    7. Send c_i^(t) to server

  Byzantine worker j:
    - Computes attack based on honest workers' RAW gradients s_h^(t)
      (not compressed deltas), then applies the same compression pipeline
      (momentum + EF21 delta + Top-k) to the attack vector and sends c_j^(t)

  Server:
    8. Update global estimates:  g_i^(t) = g_i^(t-1) + c_i^(t)  for all i
    9. Krum aggregate:  g^(t) = Krum({g_i^(t)}, f)
   10. Update model:  x^(t+1) = x^(t) - gamma * g^(t)

Key design: Byzantine attacks must see honest clients' raw gradients, not
compressed deltas.  To achieve this, the AFTER_COMPUTE hook defers writing
compressed results to context.grad for honest clients until after Byzantine
clients have computed their attacks.  The compressed deltas are stored in a
pending buffer and flushed when the first Byzantine client is processed.

The optimizer uses two hooks:
  - AFTER_COMPUTE: client-side dual momentum + EF21 error feedback + Top-k compression
                   Honest clients: compute compression but defer writing to context.grad
                   so that context.all_honest_gradients still contains raw gradients
                   when Byzantine attacks are computed.
                   Byzantine clients: apply the same compression pipeline to the
                   attack output, then flush all pending honest compressed deltas
                   into context.grad.
  - BEFORE_AGGREGATE: reconstruct per-client g_i^(t) on the server side
                      from the c_i^(t) stored in context.grad, using
                      g_i^(t) = g_i^(t-1) + c_i^(t) for ALL clients
                      (both honest and Byzantine).  All c_i^(t) are now
                      in the same singleton flat-tensor format.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch

from .base_optimizer import BaseOptimizer
from fl_framework.core.hooks import HookType, Context, hook_registry


class ByzDM21(BaseOptimizer):
    """Byzantine-robust distributed momentum with Top-k compression.

    Implements both Byz-DM21 (use_vr=False) and Byz-VR-DM21 (use_vr=True).

    Parameters
    ----------
    lr : float
        Global learning rate gamma.  Default 0.01.
    momentum : float
        Momentum coefficient eta (used as the weight for the new gradient).
        v = (1-eta)*v + eta*s, u = (1-eta)*u + eta*v.  Default 0.9.
    compression_ratio : float
        Fraction of dimensions to keep in Top-k, i.e. k = ceil(ratio * d).
        Default 0.1 (keep 10% of components).
    use_vr : bool
        If True, use variance-reduced momentum (Byz-VR-DM21):
            v_i^(t) = s_i^(t) + (1-eta)*(v_i^(t-1) - s_i^(t-1))
        If False, use standard EMA momentum (Byz-DM21):
            v_i^(t) = (1-eta)*v_i^(t-1) + eta*s_i^(t)
        Default False.
    """

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.9,
        compression_ratio: float = 0.1,
        use_vr: bool = False,
    ) -> None:
        super().__init__(lr=lr)
        self.momentum = momentum
        self.compression_ratio = compression_ratio
        self.use_vr = use_vr

        # --- Lazy-initialised state (set on first gradient) ---
        self.d: Optional[int] = None           # total gradient dimension
        self.k: Optional[int] = None           # number of components to keep
        self.reference_shapes: Optional[List[torch.Size]] = None

        # Per-client raw momentum buffers: v_i^(t)
        # Indexed by client_id; each is a flat (d,)-shaped tensor.
        self._v_buffers: List[Optional[torch.Tensor]] = []

        # Per-client auxiliary momentum buffers: u_i^(t)
        # u_i^(t) = (1-eta)*u_i^(t-1) + eta*v_i^(t)
        self._u_buffers: List[Optional[torch.Tensor]] = []

        # Per-client EF21 estimate buffers: g_i^(t-1)
        # These track the client-side estimate of each client's auxiliary momentum.
        self._g_buffers: List[Optional[torch.Tensor]] = []

        # Server-side copies of g_i^(t-1), kept in sync with client-side.
        # After compression, the server reconstructs g_i^(t) = g_i^(t-1) + c_i^(t).
        self._server_g_buffers: List[Optional[torch.Tensor]] = []

        # Per-client previous stochastic gradient: s_i^(t-1)
        # Only used when use_vr=True (Byz-VR-DM21).
        self._prev_grad_buffers: List[Optional[torch.Tensor]] = []

        # Per-client previous model parameters (flat): x^(t-1)
        # Only used when use_vr=True, to detect when the model has changed
        # so we can store the gradient at the old model.
        self._prev_model_buffers: List[Optional[torch.Tensor]] = []

        # Pending compressed deltas for honest clients.
        # Keyed by client_id; each value is a singleton list [flat_tensor]
        # containing the Top-k compressed delta c_i^(t).
        # These are computed during honest clients' AFTER_COMPUTE hook but
        # NOT written to context.grad immediately — they are deferred so
        # that context.all_honest_gradients still contains raw gradients
        # when Byzantine attacks are computed.  They are flushed into
        # context.grad when the first Byzantine client is processed.
        self._pending_compressed: dict = {}

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

        mode_str = "Byz-VR-DM21" if self.use_vr else "Byz-DM21"
        print(
            f"[{mode_str}] Initialised: d={self.d}, k={self.k}, "
            f"compression_ratio={self.compression_ratio}"
        )
        print(
            f"[{mode_str}] Top-k keeps {self.k}/{self.d} components "
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
    # Helper: get current model as flat vector
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _get_flat_model(model) -> torch.Tensor:
        """Get the current model parameters as a flat vector."""
        return torch.cat([p.data.reshape(-1) for p in model.parameters()])

    # ------------------------------------------------------------------
    # Hook 1 — AFTER_COMPUTE: client-side dual momentum + EF21 + Top-k
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _client_process(self, context: Context) -> None:
        """Per-client processing: dual momentum update, EF21 error feedback, Top-k.

        Honest clients:
          Compute the full compression pipeline (momentum → delta → Top-k),
          but DEFER writing the compressed delta to context.grad.  Instead,
          store it in self._pending_compressed.  This ensures that
          context.all_honest_gradients (set by the coordinator after all
          honest clients finish) still contains the RAW per-layer gradients,
          so Byzantine attacks can compute on the original gradients.

        Byzantine clients:
          Apply the same compression pipeline to the attack output (treating
          it as a "pseudo-gradient"), then flush all pending honest compressed
          deltas into context.grad.  This guarantees that:
          1. Byzantine attacks saw raw honest gradients
          2. All context.grad entries are in the same singleton flat-tensor
             format after this hook returns
        """
        client_id = context.current_client_id
        client = context.clients[client_id]
        client_grad = context.grad[client_id]

        # --- Lazy init on first call ---
        if self.d is None:
            self._lazy_init(client_grad, client_grad[0].device)

        # --- Ensure buffer lists are large enough ---
        while len(self._v_buffers) <= client_id:
            self._v_buffers.append(None)
        while len(self._u_buffers) <= client_id:
            self._u_buffers.append(None)
        while len(self._g_buffers) <= client_id:
            self._g_buffers.append(None)
        if self.use_vr:
            while len(self._prev_grad_buffers) <= client_id:
                self._prev_grad_buffers.append(None)
            while len(self._prev_model_buffers) <= client_id:
                self._prev_model_buffers.append(None)

        # --- Byzantine client: compress attack output, then flush pending ---
        if client.client_type == "Byzantine":
            # Apply the same compression pipeline to the attack output
            c = self._compress_gradient(client_id, client_grad)

            # Write the compressed attack delta to context.grad
            context.grad[client_id] = [c]

            # Flush all pending honest compressed deltas into context.grad
            for hid, hcompressed in self._pending_compressed.items():
                context.grad[hid] = hcompressed
            self._pending_compressed.clear()
            return

        # --- Honest client processing ---
        # Compute the full compression pipeline
        c = self._compress_gradient(client_id, client_grad)

        # DEFER: do NOT write to context.grad yet — store in pending buffer
        # so that context.all_honest_gradients still contains raw gradients
        # when Byzantine attacks are computed later.
        self._pending_compressed[client_id] = [c]

    @torch.no_grad()
    def _compress_gradient(
        self, client_id: int, client_grad: List[torch.Tensor]
    ) -> torch.Tensor:
        """Run the full compression pipeline on a gradient (raw or attack).

        This is shared by both honest and Byzantine clients:
          1. Flatten to (d,) vector
          2. Update raw momentum v_i^(t)
          3. Update auxiliary momentum u_i^(t)
          4. Compute EF21 delta: Delta_i^(t) = u_i^(t) - g_i^(t-1)
          5. Top-k compress: c_i^(t) = TopK(Delta_i^(t), k)
          6. Update local estimate: g_i^(t) = g_i^(t-1) + c_i^(t)

        Returns the compressed delta c_i^(t) as a flat (d,)-tensor.
        """
        device = client_grad[0].device
        flat_grad = self._flatten(client_grad).to(device)  # s_i^(t)

        # 1. Update raw momentum v_i^(t)
        v_buf = self._v_buffers[client_id]
        if v_buf is None:
            # First iteration: v_i^(0) = s_i^(0)
            v_buf = flat_grad.clone()
        else:
            v_buf = v_buf.to(device)
            if self.use_vr:
                # Byz-VR-DM21: v_i^(t) = s_i^(t) + (1-eta)*(v_i^(t-1) - s_i^(t-1))
                prev_grad = self._prev_grad_buffers[client_id]
                if prev_grad is None:
                    # First step with VR: no previous gradient, fall back to
                    # v_i^(0) = s_i^(0) (same as non-VR initialization)
                    v_buf = flat_grad.clone()
                else:
                    prev_grad = prev_grad.to(device)
                    # v = s_new + (1-eta)*(v_old - s_old)
                    v_buf = flat_grad + (1.0 - self.momentum) * (v_buf - prev_grad)
            else:
                # Byz-DM21: v_i^(t) = (1-eta)*v_i^(t-1) + eta*s_i^(t)
                v_buf.mul_(1.0 - self.momentum).add_(flat_grad, alpha=self.momentum)
        self._v_buffers[client_id] = v_buf

        # Store current gradient as previous for next step (VR mode)
        if self.use_vr:
            self._prev_grad_buffers[client_id] = flat_grad.clone()

        # 2. Update auxiliary momentum: u_i^(t) = (1-eta)*u_i^(t-1) + eta*v_i^(t)
        u_buf = self._u_buffers[client_id]
        if u_buf is None:
            # First iteration: u_i^(0) = v_i^(0)
            u_buf = v_buf.clone()
        else:
            u_buf = u_buf.to(device)
            u_buf.mul_(1.0 - self.momentum).add_(v_buf, alpha=self.momentum)
        self._u_buffers[client_id] = u_buf

        # 3. Compute delta: Delta_i^(t) = u_i^(t) - g_i^(t-1)
        g_buf = self._g_buffers[client_id]
        if g_buf is None:
            # g_i^(-1) = 0, so Delta = u
            delta = u_buf.clone()
        else:
            g_buf = g_buf.to(device)
            delta = u_buf - g_buf

        # 4. Top-k compress: c_i^(t) = TopK(Delta_i^(t), k)
        c = self._topk(delta, self.k)

        # 5. Update local estimate: g_i^(t) = g_i^(t-1) + c_i^(t)
        if g_buf is None:
            g_buf = c.clone()
        else:
            g_buf = g_buf.to(device)
            g_buf.add_(c)
        self._g_buffers[client_id] = g_buf

        return c

    # ------------------------------------------------------------------
    # Hook 2 — BEFORE_AGGREGATE: server-side reconstruction of g_i^(t)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _server_reconstruct(self, context: Context) -> None:
        """Reconstruct each client's g_i^(t) on the server side.

        First, flush any pending honest compressed deltas into context.grad.
        This handles the case where num_byzantine=0 (no Byzantine clients
        to trigger the flush in _client_process).

        Then, for ALL clients (both honest and Byzantine), context.grad[i]
        contains the compressed delta c_i^(t) as a singleton list [flat_tensor]
        with shape (d,).  The server maintains g_i^(t-1) and reconstructs:
            g_i^(t) = g_i^(t-1) + c_i^(t)

        After reconstruction, each context.grad[i] is replaced with the
        full g_i^(t) vector (as a singleton list) so that the Krum
        aggregator can operate on the reconstructed estimates.
        """
        # Flush any pending honest compressed deltas (needed when num_byzantine=0)
        if self._pending_compressed:
            for hid, hcompressed in self._pending_compressed.items():
                context.grad[hid] = hcompressed
            self._pending_compressed.clear()

        for i in range(len(context.grad)):
            g = context.grad[i]
            if g is None:
                continue

            # Ensure server-side buffer list is large enough
            while len(self._server_g_buffers) <= i:
                self._server_g_buffers.append(None)

            # c_i^(t) is a singleton list [flat (d,)-tensor] for ALL clients
            # (both honest and Byzantine, since Byzantine attack output now
            # also goes through the compression pipeline)
            c_i = g[0]

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
            mode_str = "Byz-VR-DM21" if self.use_vr else "Byz-DM21"
            print(
                f"[{mode_str}] Warning: server or aggregated_grad is None, "
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
            "use_vr": self.use_vr,
            "d": self.d,
            "k": self.k,
            "reference_shapes": self.reference_shapes,
        }

        # v buffers — move to CPU for serialisation.
        if self._v_buffers:
            state["v_buffers"] = [
                buf.cpu().clone() if buf is not None else None
                for buf in self._v_buffers
            ]

        # u buffers
        if self._u_buffers:
            state["u_buffers"] = [
                buf.cpu().clone() if buf is not None else None
                for buf in self._u_buffers
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

        # Previous gradient buffers (VR mode)
        if self.use_vr and self._prev_grad_buffers:
            state["prev_grad_buffers"] = [
                buf.cpu().clone() if buf is not None else None
                for buf in self._prev_grad_buffers
            ]

        return state

    def set_state(self, state: dict) -> None:
        """Restore state from a checkpoint."""
        self.lr = state.get("lr", self.lr)
        self.momentum = state.get("momentum", self.momentum)
        self.compression_ratio = state.get("compression_ratio", self.compression_ratio)
        self.use_vr = state.get("use_vr", self.use_vr)
        self.d = state.get("d")
        self.k = state.get("k")
        self.reference_shapes = state.get("reference_shapes")

        raw_v = state.get("v_buffers")
        if raw_v is not None:
            self._v_buffers = [
                buf.clone() if buf is not None else None for buf in raw_v
            ]

        raw_u = state.get("u_buffers")
        if raw_u is not None:
            self._u_buffers = [
                buf.clone() if buf is not None else None for buf in raw_u
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

        if self.use_vr:
            raw_pg = state.get("prev_grad_buffers")
            if raw_pg is not None:
                self._prev_grad_buffers = [
                    buf.clone() if buf is not None else None for buf in raw_pg
                ]
