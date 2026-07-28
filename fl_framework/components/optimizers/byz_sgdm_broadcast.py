# fl_framework/components/optimizers/byz_sgdm_broadcast.py
"""ByzSGDMBroadcast: Byzantine-robust SGDM with rand-k compression and Krum aggregation.

Combines:
  - Client-side stochastic gradient descent with momentum (SGDM)
  - BROADCAST-style error feedback via auxiliary vectors h_omega
  - Unbiased rand-k sparsification (compress the *difference* g_omega - h_omega)
  - Krum-based robust aggregation on the server (delegated to the aggregator)

Algorithm outline (per round t):

  Honest worker omega:
    1. Sample a mini-batch of B' local samples
    2. Compute stochastic gradient:  grad = (1/B') * sum_{i in batch} grad f_{omega,i}(x^t)
    3. Update momentum:  m_omega^t = mu * m_omega^{t-1} + grad
    4. Set g_omega^t = m_omega^t  (use momentum-smoothed direction)
    5. Compute difference:  u_omega^t = g_omega^t - h_omega^t
    6. Rand-k compress:  c_hat_omega^t = RandKCompress(u_omega^t, k)
    7. Send c_hat_omega^t to master
    8. Update local auxiliary vector:  h_omega^{t+1} = h_omega^t + beta * c_hat_omega^t

  Byzantine worker omega:
    - Computes attack based on honest workers' RAW gradients (not compressed deltas),
      then applies the same compression pipeline (momentum + BROADCAST delta + rand-k)
      to the attack vector and sends c_hat_omega^t.

  Server (master):
    9. Reconstruct approximate gradient for each worker:
           g_hat_omega^t = h_omega^t + c_hat_omega^t
   10. Krum aggregate:  z^t = Krum({g_hat_1^t, ..., g_hat_W^t}, f)
   11. Update global model:  x^{t+1} = x^t - gamma * z^t
   12. Update master-side auxiliary vectors (keep in sync with workers):
           h_omega^{t+1} = h_omega^t + beta * c_hat_omega^t

Key design: Byzantine attacks must see honest clients' raw gradients, not
compressed deltas.  To achieve this, the AFTER_COMPUTE hook defers writing
compressed results to context.grad for honest clients until after Byzantine
clients have computed their attacks.  The compressed deltas are stored in a
pending buffer and flushed when the first Byzantine client is processed.

Symbol glossary:
  t       - iteration round
  x^t     - global model parameters at round t
  gamma   - global learning rate, fixed at 0.1
  mu      - momentum coefficient (typically 0.9)
  beta    - compression step size for auxiliary vector update. Must satisfy
            beta * (1 + delta) <= 1 for convergence guarantee. Default 0.1.
  B'      - mini-batch size for local stochastic gradient
  p       - model dimension (gradient vector length)
  k       - number of coordinates kept by rand-k; k = compression_ratio * p
  delta   - compressor variance parameter; for rand-k, delta = p/k - 1
  f       - known upper bound on Byzantine workers; must satisfy f < (W - 2)
  R       - set of regular (honest) workers
  B       - set of Byzantine workers
  W       - set of all workers, W = R union B
  m_omega^t       - momentum buffer on worker omega
  g_omega^t       - momentum-smoothed gradient direction (= m_omega^t)
  h_omega^t       - auxiliary vector for worker omega (kept on both worker and master)
  u_omega^t       - difference to compress: g_omega^t - h_omega^t
  C(v)            - unbiased rand-k compressor: randomly keep k coords, scale by p/k
  c_hat_omega^t   - compressed difference C(g_omega^t - h_omega^t)
  g_hat_omega^t   - master-reconstructed approximate gradient: h_omega^t + c_hat_omega^t
  Krum({...}, f)  - Krum robust aggregation operator

The optimizer uses two hooks:
  - AFTER_COMPUTE: client-side momentum update + rand-k compression
                   Honest clients: compute compression but defer writing to context.grad
                   so that context.all_honest_gradients still contains raw gradients
                   when Byzantine attacks are computed.
                   Byzantine clients: apply the same compression pipeline to the
                   attack output, then flush all pending honest compressed deltas
                   into context.grad.
  - BEFORE_AGGREGATE: server-side reconstruction of g_hat_omega^t
                      from c_hat_omega^t stored in context.grad, using
                      g_hat_omega^t = h_omega^t + c_hat_omega^t for ALL clients
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch

from .base_optimizer import BaseOptimizer
from fl_framework.core.hooks import HookType, Context, hook_registry


class ByzSGDMBroadcast(BaseOptimizer):
    """Byzantine-robust SGDM with rand-k compression and Krum aggregation.

    Parameters
    ----------
    lr : float
        Global learning rate gamma. Default 0.1.
    momentum : float
        Momentum coefficient mu. Default 0.9.
    beta : float
        Compression step size for auxiliary vector update. Controls how much
        the new compressed difference influences h_omega. Must satisfy
        beta * (1 + delta) <= 1 for convergence guarantee. Default 0.1.
    compression_ratio : float
        Fraction of dimensions to keep in rand-k, i.e. k = ceil(ratio * d).
        Default 0.1 (keep 10% of components).
    weight_decay : float
        L2 regularization factor. Default 0.0.
    """

    def __init__(
        self,
        lr: float = 0.1,
        momentum: float = 0.9,
        beta: float = 0.1,
        compression_ratio: float = 0.1,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(lr=lr)
        self.momentum = momentum
        self.beta = beta
        self.compression_ratio = compression_ratio
        self.weight_decay = weight_decay

        # --- Lazy-initialised state (set on first gradient) ---
        self.d: Optional[int] = None           # total gradient dimension (p)
        self.k: Optional[int] = None           # number of coordinates to keep
        self.reference_shapes: Optional[List[torch.Size]] = None

        # Per-client momentum buffers: m_omega^t
        # Indexed by client_id; each is a flat (d,)-shaped tensor.
        self._momentum_buffers: List[Optional[torch.Tensor]] = []

        # Per-client auxiliary vectors: h_omega^t (client-side copies)
        # Indexed by client_id; each is a flat (d,)-shaped tensor.
        self._h_buffers: List[Optional[torch.Tensor]] = []

        # Server-side copies of h_omega^t, kept in sync with client-side.
        # After compression, the server reconstructs g_hat_omega^t = h_omega^t + c_hat_omega^t.
        self._server_h_buffers: List[Optional[torch.Tensor]] = []

        # Pending compressed deltas for honest clients.
        # Keyed by client_id; each value is a singleton list [flat_tensor]
        # containing the rand-k compressed delta c_hat_omega^t.
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

        delta = self.d / self.k - 1  # rand-k variance parameter
        print(
            f"[ByzSGDMBroadcast] Initialised: d={self.d}, k={self.k}, "
            f"compression_ratio={self.compression_ratio}, delta={delta:.1f}"
        )
        print(
            f"[ByzSGDMBroadcast] Rand-k keeps {self.k}/{self.d} components "
            f"({self.k / self.d * 100:.1f}%)"
        )
        print(
            f"[ByzSGDMBroadcast] lr={self.lr}, momentum={self.momentum}, beta={self.beta}, "
            f"beta*(1+delta)={self.beta * (1 + delta):.4f} (must be <= 1 for convergence)"
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
    # Rand-k compression (unbiased)
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _randk_compress(vec: torch.Tensor, k: int) -> torch.Tensor:
        """Unbiased rand-k sparsification: randomly keep k coordinates,
        scale each by d/k to ensure unbiasedness, set the rest to zero.

        E[C(v)] = v  (unbiased property)

        Parameters
        ----------
        vec : torch.Tensor
            Flat (d,)-shaped input vector.
        k : int
            Number of coordinates to keep.

        Returns
        -------
        torch.Tensor
            Compressed vector of the same shape as *vec*.
        """
        d = vec.numel()
        if k >= d:
            return vec.clone()

        # Randomly select k distinct indices
        indices = torch.randperm(d, device=vec.device)[:k]

        # Scale factor for unbiasedness: p/k
        scale = d / k

        result = torch.zeros_like(vec)
        result[indices] = vec[indices] * scale
        return result

    # ------------------------------------------------------------------
    # Shared compression pipeline
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compress_gradient(
        self, client_id: int, client_grad: List[torch.Tensor], client
    ) -> torch.Tensor:
        """Run the full compression pipeline on a gradient (raw or attack).

        This is shared by both honest and Byzantine clients:
          1. Flatten to (d,) vector
          2. Apply weight decay (honest clients only, using client's model)
          3. Update momentum: m_omega^t = mu * m_omega^{t-1} + grad
          4. Set g_omega^t = m_omega^t
          5. Compute difference: u_omega^t = g_omega^t - h_omega^t
          6. Rand-k compress: c_hat_omega^t = RandKCompress(u_omega^t, k)
          7. Update local auxiliary vector: h_omega^{t+1} = h_omega^t + beta * c_hat_omega^t

        Returns the compressed delta c_hat_omega^t as a flat (d,)-tensor.
        """
        device = client_grad[0].device
        flat_grad = self._flatten(client_grad).to(device)

        # Apply weight decay to the gradient (honest clients only)
        if self.weight_decay > 0 and client.client_type == "Honest":
            model = client.model
            param_flat = self._flatten([p.data for p in model.parameters()]).to(device)
            flat_grad.add_(param_flat, alpha=self.weight_decay)

        # 1. Update momentum: m_omega^t = mu * m_omega^{t-1} + grad
        m_buf = self._momentum_buffers[client_id]
        if m_buf is None:
            # First iteration: m_omega^0 = grad
            m_buf = flat_grad.clone()
        else:
            m_buf = m_buf.to(device)
            m_buf.mul_(self.momentum).add_(flat_grad)
        self._momentum_buffers[client_id] = m_buf

        # 2. g_omega^t = m_omega^t (momentum-smoothed direction)
        g_omega = m_buf

        # 3. Compute difference: u_omega^t = g_omega^t - h_omega^t
        h_buf = self._h_buffers[client_id]
        if h_buf is None:
            h_buf = torch.zeros(self.d, device=device)
        else:
            h_buf = h_buf.to(device)
        u = g_omega - h_buf

        # 4. Rand-k compress: c_hat_omega^t = RandKCompress(u_omega^t, k)
        c_hat = self._randk_compress(u, self.k)

        # 5. Update local auxiliary vector: h_omega^{t+1} = h_omega^t + beta * c_hat_omega^t
        h_buf_new = h_buf + self.beta * c_hat
        self._h_buffers[client_id] = h_buf_new.cpu()

        return c_hat

    # ------------------------------------------------------------------
    # Hook 1 — AFTER_COMPUTE: client-side momentum + rand-k compression
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _client_process(self, context: Context) -> None:
        """Per-client processing: momentum update + rand-k compression.

        Honest clients:
          Compute the full compression pipeline (momentum → delta → rand-k),
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
        while len(self._momentum_buffers) <= client_id:
            self._momentum_buffers.append(None)
        while len(self._h_buffers) <= client_id:
            self._h_buffers.append(None)

        # --- Byzantine client: compress attack output, then flush pending ---
        if client.client_type == "Byzantine":
            # Apply the same compression pipeline to the attack output
            c_hat = self._compress_gradient(client_id, client_grad, client)

            # Write the compressed attack delta to context.grad
            context.grad[client_id] = [c_hat]

            # Flush all pending honest compressed deltas into context.grad
            for hid, hcompressed in self._pending_compressed.items():
                context.grad[hid] = hcompressed
            self._pending_compressed.clear()
            return

        # --- Honest client processing ---
        # Compute the full compression pipeline
        c_hat = self._compress_gradient(client_id, client_grad, client)

        # DEFER: do NOT write to context.grad yet — store in pending buffer
        # so that context.all_honest_gradients still contains raw gradients
        # when Byzantine attacks are computed later.
        self._pending_compressed[client_id] = [c_hat]

    # ------------------------------------------------------------------
    # Hook 2 — BEFORE_AGGREGATE: server-side reconstruction of g_hat_omega^t
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _server_reconstruct(self, context: Context) -> None:
        """Reconstruct each client's g_hat_omega^t on the server side.

        First, flush any pending honest compressed deltas into context.grad.
        This handles the case where num_byzantine=0 (no Byzantine clients
        to trigger the flush in _client_process).

        Then, for ALL clients (both honest and Byzantine), context.grad[i]
        contains the compressed delta c_hat_omega^t as a singleton list
        [flat_tensor] with shape (d,).  The server maintains h_omega^t and
        reconstructs:
            g_hat_omega^t = h_omega^t + c_hat_omega^t

        Then updates the server-side auxiliary vector:
            h_omega^{t+1} = h_omega^t + beta * c_hat_omega^t

        After reconstruction, each context.grad[i] is replaced with the
        full g_hat_omega^t vector (as a singleton list) so that the Krum
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
            while len(self._server_h_buffers) <= i:
                self._server_h_buffers.append(None)

            # c_hat_omega^t is a singleton list [flat (d,)-tensor] for ALL clients
            # (both honest and Byzantine, since Byzantine attack output now
            # also goes through the compression pipeline)
            c_i = g[0]

            # Reconstruct: g_hat_omega^t = h_omega^t + c_hat_omega^t
            device = c_i.device
            server_h = self._server_h_buffers[i]
            if server_h is None:
                # h_omega^0 = 0, so g_hat_omega^t = c_hat_omega^t
                g_hat = c_i.clone()
            else:
                server_h = server_h.to(device)
                g_hat = server_h + c_i

            # Update server-side auxiliary vector: h_omega^{t+1} = h_omega^t + beta * c_hat_omega^t
            if server_h is None:
                server_h_new = self.beta * c_i
            else:
                server_h_new = server_h + self.beta * c_i
            self._server_h_buffers[i] = server_h_new.cpu()

            # Replace with the reconstructed g_hat_omega^t
            context.grad[i] = [g_hat]

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
            selected by Krum (the winning client's g_hat_omega^t).
        """
        if server is None or aggregated_grad is None:
            print(
                "[ByzSGDMBroadcast] Warning: server or aggregated_grad is None, "
                "skipping update."
            )
            return

        # Krum returns [flat_vector] — the winning client's g_hat_omega^t.
        g_aggregated = aggregated_grad[0]  # shape (d,)

        device = next(server.model.parameters()).device
        g_aggregated = g_aggregated.to(device, non_blocking=True)

        # Reshape to per-parameter tensors and apply update: x^{t+1} = x^t - gamma * z^t
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
            "beta": self.beta,
            "compression_ratio": self.compression_ratio,
            "weight_decay": self.weight_decay,
            "d": self.d,
            "k": self.k,
            "reference_shapes": self.reference_shapes,
        }

        # Momentum buffers — move to CPU for serialisation
        if self._momentum_buffers:
            state["momentum_buffers"] = [
                buf.cpu().clone() if buf is not None else None
                for buf in self._momentum_buffers
            ]

        # Client-side h buffers
        if self._h_buffers:
            state["h_buffers"] = [
                buf.cpu().clone() if buf is not None else None
                for buf in self._h_buffers
            ]

        # Server-side h buffers
        if self._server_h_buffers:
            state["server_h_buffers"] = [
                buf.cpu().clone() if buf is not None else None
                for buf in self._server_h_buffers
            ]

        return state

    def set_state(self, state: dict) -> None:
        """Restore state from a checkpoint."""
        self.lr = state.get("lr", self.lr)
        self.momentum = state.get("momentum", self.momentum)
        self.beta = state.get("beta", self.beta)
        self.compression_ratio = state.get("compression_ratio", self.compression_ratio)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        self.d = state.get("d")
        self.k = state.get("k")
        self.reference_shapes = state.get("reference_shapes")

        raw_mom = state.get("momentum_buffers")
        if raw_mom is not None:
            self._momentum_buffers = [
                buf.clone() if buf is not None else None for buf in raw_mom
            ]

        raw_h = state.get("h_buffers")
        if raw_h is not None:
            self._h_buffers = [
                buf.clone() if buf is not None else None for buf in raw_h
            ]

        raw_sh = state.get("server_h_buffers")
        if raw_sh is not None:
            self._server_h_buffers = [
                buf.clone() if buf is not None else None for buf in raw_sh
            ]
