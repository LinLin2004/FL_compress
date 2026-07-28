# fl_framework/components/optimizers/fed_dproc.py
"""FedDPRoC: Federated Differentially Private Robust Compressed optimization.

Combines Count-Sketch compression, differential privacy noise, client-side
momentum, and Krum-based robust aggregation in a single optimizer.

The optimizer uses two hooks:
  - AFTER_COMPUTE: lazy initialisation only. Gradients are deliberately left
                   untouched here so attacks read raw honest gradients from
                   context.all_honest_gradients.
  - BEFORE_AGGREGATE: gradient clipping, Gaussian noise, momentum update for
                      honest clients, then Count-Sketch compression for ALL
                      clients.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch

from .base_optimizer import BaseOptimizer
from fl_framework.core.hooks import HookType, Context, hook_registry


class FedDPRoC(BaseOptimizer):
    """Federated learning optimizer with Count-Sketch compression and DP noise.

    Parameters
    ----------
    lr : float
        Global learning rate.
    alpha : float
        Compression ratio d/k.  E.g. alpha=4 means the compressed dimension
        is roughly d/4.  Default 4.
    p : int
        Number of Count-Sketch blocks.  Must divide k.  Default 8.
    beta : float
        Momentum coefficient.  Default 0.9.
    clip_threshold : float
        Gradient clipping threshold C.  Default 10.0.
    noise_multiplier : float
        Gaussian noise multiplier sigma_NM.  The noise std applied to each
        gradient is 2 * C * sigma_NM.  Set to 0 to disable noise.  Default 0.01.
    seed : int
        Random seed for reproducible Count-Sketch hash generation.  Default 42.
    """

    def __init__(
        self,
        lr: float,
        alpha: float = 4.0,
        p: int = 8,
        beta: float = 0.9,
        clip_threshold: float = 10.0,
        noise_multiplier: float = 0.01,
        seed: int = 42,
    ) -> None:
        super().__init__(lr=lr)
        self.alpha = alpha
        self.p = p
        self.beta = beta
        self.clip_threshold = clip_threshold
        self.noise_multiplier = noise_multiplier
        self.seed = seed

        # --- Lazy-initialised state (set on first gradient) ---
        self.d: Optional[int] = None           # total gradient dimension
        self.k: Optional[int] = None           # compressed dimension (multiple of p)
        self.s: Optional[int] = None           # rows per block = k / p
        self.reference_shapes: Optional[List[torch.Size]] = None

        # Count-Sketch hash tables (stored on CPU to save GPU memory)
        self.hash_indices: Optional[List[torch.Tensor]] = None
        self.signs: Optional[List[torch.Tensor]] = None

        # Per-client momentum buffers: momentum_buffers[client_id] is a flat
        # (d,)-shaped tensor on the client's device, or None if not yet created.
        self.momentum_buffers: List[Optional[torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def register_hooks(self) -> None:
        super().register_hooks()
        hook_registry.register(HookType.AFTER_COMPUTE, self._client_process)
        hook_registry.register(HookType.BEFORE_AGGREGATE, self._compress_all)

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _lazy_init(self, sample_grad: List[torch.Tensor], device: torch.device) -> None:
        """Initialise dimensions, Count-Sketch tables and momentum buffers.

        Called once on the first AFTER_COMPUTE trigger so that ``d`` (total
        number of scalar parameters) is known.
        """
        self.reference_shapes = [g.shape for g in sample_grad]
        flat = self._flatten(sample_grad)
        self.d = flat.numel()

        # Compression dimension k = floor(d / alpha), rounded down to
        # a multiple of p (and at least p so that s >= 1).
        self.k = max(self.p, int(self.d / self.alpha))
        self.k = (self.k // self.p) * self.p
        self.s = self.k // self.p

        print(
            f"[FedDPRoC] Initialised: d={self.d}, k={self.k}, s={self.s}, p={self.p}"
        )
        print(
            f"[FedDPRoC] Compression ratio: {self.d}/{self.k} ≈ "
            f"{self.k / self.d * 100:.1f}%"
        )

        # Build Count-Sketch hash tables on CPU (deterministic, reproducible).
        cpu_gen = torch.Generator(device="cpu").manual_seed(self.seed)
        self.hash_indices = []
        self.signs = []
        for _ in range(self.p):
            idx = torch.randint(
                0, self.s, (self.d,), generator=cpu_gen, dtype=torch.long
            )
            sign = (
                torch.randint(
                    0, 2, (self.d,), generator=cpu_gen, dtype=torch.float32
                )
                * 2
                - 1
            )
            self.hash_indices.append(idx)   # stays on CPU
            self.signs.append(sign)          # stays on CPU

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
            result.append(flat[offset : offset + n].view(shape))
            offset += n
        return result

    # ------------------------------------------------------------------
    # Count-Sketch core:  y = R @ v   and   x = R^T @ y
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compress(self, vec: torch.Tensor) -> torch.Tensor:
        """Count-Sketch forward pass:  R @ vec  — (d,) → (k,).

        Each of the *p* blocks independently hashes *d* input positions into
        *s* buckets with a random sign, then the block outputs are
        concatenated and scaled by 1/√p.
        """
        device = vec.device
        result = torch.zeros(self.k, device=device, dtype=vec.dtype)

        for b in range(self.p):
            # Move one block's tables to GPU on demand.
            idx = self.hash_indices[b].to(device, non_blocking=True)
            sign = self.signs[b].to(device, non_blocking=True)

            # Bucket accumulation:  out[j] = Σ_{l: h(l)=j} sign(l) * vec[l]
            block_out = torch.zeros(self.s, device=device, dtype=vec.dtype)
            block_out.scatter_add_(0, idx, sign * vec)

            start = b * self.s
            result[start : start + self.s] = block_out

        result.div_(math.sqrt(self.p))
        return result

    @torch.no_grad()
    def _decompress(self, compressed: torch.Tensor) -> torch.Tensor:
        """Count-Sketch inverse:  R^T @ compressed  — (k,) → (d,).

        For each block, the block's *s* bucket values are gathered back to
        *d* positions using the same hash indices, multiplied by the
        corresponding sign, and accumulated across blocks.
        """
        device = compressed.device
        result = torch.zeros(self.d, device=device, dtype=compressed.dtype)

        for b in range(self.p):
            idx = self.hash_indices[b].to(device, non_blocking=True)
            sign = self.signs[b].to(device, non_blocking=True)

            start = b * self.s
            block_vals = compressed[start : start + self.s]

            # result[l] += sign(l) * compressed[ h(l) ]
            result.add_(sign * block_vals[idx])

        result.div_(math.sqrt(self.p))
        return result

    # ------------------------------------------------------------------
    # Hook 1 — AFTER_COMPUTE: lazy init only
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _client_process(self, context: Context) -> None:
        """Initialise optimizer state without modifying the computed gradient.

        The coordinator snapshots ``context.all_honest_gradients`` after all
        honest clients complete their ``AFTER_COMPUTE`` hooks.  Therefore this
        hook must not overwrite ``context.grad``; otherwise FOE-like attacks
        would be based on clipped/noised/momentum gradients instead of raw
        backward gradients.
        """
        client_id = context.current_client_id
        client_grad = context.grad[client_id]

        # --- Lazy init on first call ---
        if self.d is None:
            self._lazy_init(client_grad, client_grad[0].device)

        # --- Ensure momentum buffer list is large enough ---
        while len(self.momentum_buffers) <= client_id:
            self.momentum_buffers.append(None)

    # ------------------------------------------------------------------
    # Deferred honest-client processing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _process_honest_gradient(
        self,
        client_id: int,
        client_grad: List[torch.Tensor],
        client,
    ) -> List[torch.Tensor]:
        """Apply clipping, optional DP noise and momentum to one honest gradient."""
        device = client_grad[0].device
        flat = self._flatten(client_grad).to(device)

        # 1. Batch-level gradient clipping
        grad_norm = torch.norm(flat)
        clip_coeff = min(1.0, self.clip_threshold / (grad_norm + 1e-8))
        flat.mul_(clip_coeff)

        # 2. Gaussian noise
        if self.noise_multiplier > 0.0:
            # Try to read batch size from the sampler for correct DP scaling.
            batch_size = getattr(
                getattr(client, "sampler", None), "batch_size", 1
            )
            noise_std = 2.0 * self.clip_threshold * self.noise_multiplier / batch_size
            noise = torch.randn(self.d, device=device, dtype=flat.dtype)
            noise.mul_(noise_std)
            flat.add_(noise)

        # 3. Client-side momentum:  m = beta * m + (1 - beta) * g~
        buf = self.momentum_buffers[client_id]
        if buf is None:
            buf = torch.zeros(self.d, device=device, dtype=flat.dtype)
            self.momentum_buffers[client_id] = buf
        else:
            # Move buffer to the correct device if needed (e.g. after resume).
            buf = buf.to(device)
            self.momentum_buffers[client_id] = buf

        buf.mul_(self.beta).add_(flat, alpha=(1.0 - self.beta))

        return self._unflatten(buf)

    # ------------------------------------------------------------------
    # Hook 2 — BEFORE_AGGREGATE: process honest clients, then compress all
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compress_all(self, context: Context) -> None:
        """Process honest gradients, then compress every client's message.

        Honest gradients are processed here instead of in ``AFTER_COMPUTE`` so
        Byzantine attacks have already consumed raw honest gradients.

        Each ``context.grad[i]`` is replaced by ``[k_tensor]`` — a singleton
        list so the interface ``List[List[Tensor]]`` expected by aggregators
        remains valid.
        """
        for i in range(len(context.grad)):
            g = context.grad[i]
            if g is None:
                continue

            client = context.clients[i]
            if client.client_type != "Byzantine":
                g = self._process_honest_gradient(i, g, client)
                context.grad[i] = g

            flat = self._flatten(g)                   # (d,)
            compressed = self._compress(flat)          # (k,)
            context.grad[i] = [compressed]

    # ------------------------------------------------------------------
    # Server-side update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, server, aggregated_grad) -> None:
        """Decompress the Krum-selected gradient and update the global model.

        Parameters
        ----------
        server : Server
            The central server holding the global model.
        aggregated_grad : List[Tensor]
            A singleton list containing the (k,)-shaped compressed vector
            selected by Krum.
        """
        if server is None or aggregated_grad is None:
            print(
                "[FedDPRoC] Warning: server or aggregated_grad is None, "
                "skipping update."
            )
            return

        # Krum returns [k_tensor] — the winning client's compressed vector.
        compressed = aggregated_grad[0]  # shape (k,)

        device = next(server.model.parameters()).device
        compressed = compressed.to(device, non_blocking=True)

        # Decompress: (k,) → (d,)
        decompressed = self._decompress(compressed)

        # Reshape to per-parameter tensors and apply update
        updates = self._unflatten(decompressed)

        for param, update in zip(server.model.parameters(), updates):
            param.data.add_(update.to(param.device), alpha=-self.lr)

    # ------------------------------------------------------------------
    # State persistence (checkpoint / resume)
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return serialisable state for checkpointing."""
        state: dict = {
            "lr": self.lr,
            "alpha": self.alpha,
            "p": self.p,
            "beta": self.beta,
            "clip_threshold": self.clip_threshold,
            "noise_multiplier": self.noise_multiplier,
            "seed": self.seed,
            "d": self.d,
            "k": self.k,
            "s": self.s,
            "reference_shapes": self.reference_shapes,
        }

        # Hash tables are already on CPU; store directly.
        if self.hash_indices is not None:
            state["hash_indices"] = self.hash_indices
            state["signs"] = self.signs

        # Momentum buffers — move to CPU for serialisation.
        if self.momentum_buffers:
            state["momentum_buffers"] = [
                buf.cpu().clone() if buf is not None else None
                for buf in self.momentum_buffers
            ]

        return state

    def set_state(self, state: dict) -> None:
        """Restore state from a checkpoint."""
        self.lr = state.get("lr", self.lr)
        self.alpha = state.get("alpha", self.alpha)
        self.p = state.get("p", self.p)
        self.beta = state.get("beta", self.beta)
        self.clip_threshold = state.get("clip_threshold", self.clip_threshold)
        self.noise_multiplier = state.get("noise_multiplier", self.noise_multiplier)
        self.seed = state.get("seed", self.seed)
        self.d = state.get("d")
        self.k = state.get("k")
        self.s = state.get("s")
        self.reference_shapes = state.get("reference_shapes")
        self.hash_indices = state.get("hash_indices")
        self.signs = state.get("signs")

        raw_bufs = state.get("momentum_buffers")
        if raw_bufs is not None:
            self.momentum_buffers = [
                buf.clone() if buf is not None else None for buf in raw_bufs
            ]
