# fl_framework/components/optimizers/rosdhb.py
"""RoSDHB: Robust Sparsified Distributed Heavy-Ball method.

Implements the algorithm of arXiv 2508.17129 (Algorithm 1).  RoSDHB combines:

  - A *global random sparsification mask*: every round the server picks k of the
    d coordinates (uniformly without replacement).  All honest workers apply the
    *same* mask and upload only those k scalars — the upload cost is k floats
    per worker regardless of d.
  - *Unbiased reconstruction* on the server: each worker's k values are placed
    back into a d-vector at the masked positions and scaled by d/k, so that
    E[g_tilde_i] = grad L_i(theta).  Unselected coordinates are 0.
  - *Server-side Heavy-Ball momentum*: the server keeps a momentum buffer m_i
    for every worker (honest + byzantine) and updates it from the *reconstructed*
    compressed gradient:
        m_i <- beta * m_i + (1 - beta) * g_tilde_i
    Workers store no extra state — memory is halved compared with worker-side
    momentum schemes such as Byz-DASHA-PAGE.
  - A *robust aggregator* F applied to the momentum set:
        R <- F(m_1, ..., m_n)
    Here F is delegated to the framework's aggregator (KrumAggregator), which is
    an (f, kappa)-robust aggregator tolerating up to f byzantine vectors.
  - *Heavy-Ball model update*:
        theta <- theta - gamma * R

Algorithm outline (per round t):

  Server:
    1. Generate the global sparsification mask: pick k distinct indices
       {l_1, ..., l_k} subset of {1, ..., d}.  A shared PRNG seed keeps server
       and workers in sync without transmitting the mask.
    2. Broadcast theta and mask to all workers.

  Honest worker i:
    3a. Compute local gradient  g_i <- grad L_i(theta)
    3b. Compress: C_k(g_i) <- (g_i[l_1], ..., g_i[l_k])   (length-k vector)
    3c. Send the k values to the server.

  Byzantine worker j:
    - Observes the honest workers' raw gradients in this simulator and
      constructs an attack vector from those uncompressed, unmodified values.
      Compression is applied only after Byzantine attacks have finished.

  Server:
    4. Unbiased reconstruction (per worker i):
        g_tilde_i <- 0 in R^d
        g_tilde_i[l_j] <- (d/k) * (the j-th value worker i sent),  j = 1..k
    5. Server-side momentum update (per worker i):
        m_i <- beta * m_i + (1 - beta) * g_tilde_i
    6. Robust aggregation:
        R <- F(m_1, ..., m_n)        # Krum, tolerating <= f byzantine vectors
    7. Heavy-Ball model update:
        theta <- theta - gamma * R

The optimizer uses two hooks:
  - AFTER_COMPUTE: lazy initialisation and mask preparation only.  It does not
    modify context.grad, so the coordinator sets context.all_honest_gradients
    to the honest workers' raw backward gradients.
  - BEFORE_AGGREGATE: the server reconstructs g_tilde_i for every worker
    (honest + byzantine) after applying the global mask compression to raw
    full-gradient messages, updates the per-worker momentum buffer m_i, and
    replaces context.grad[i] with [m_i] so the Krum aggregator sees the
    momentum set.

Implementation notes (avoiding ambiguity):
  * The global mask is generated *per round* in BEFORE_ROUND_SERVER and stored
    in context.extra so the BEFORE_AGGREGATE hook uses a single mask for all
    clients.  On the first round, the mask is created lazily after the first
    gradient reveals d.
  * Unbiasedness: scaling the masked coordinates by d/k makes
    E[g_tilde_i] = g_i exactly, with the mask drawn uniformly without
    replacement.  The variance is controlled by k (larger k -> less noise).
  * Momentum on the *server*: m_i is updated from g_tilde_i (which already
    carries compression noise).  The Heavy-Ball parameter beta (e.g. 0.8)
    smooths this noise; theoretically beta = sqrt(1 - 24*gamma*L).
  * Robust aggregation is delegated to the configured aggregator (Krum).  Krum
    returns one client's momentum vector, which is exactly R = F(m_1..m_n).
  * Byzantine attack compatibility: context.all_honest_gradients contains the
    original per-layer gradients, so FOE/ALIE/Mimic-style attacks compute from
    raw honest gradients.  Their forged full-gradient outputs are then passed
    through the same RoSDHB compression/reconstruction path before momentum
    and aggregation.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch

from .base_optimizer import BaseOptimizer
from fl_framework.core.hooks import HookType, Context, hook_registry


class RoSDHB(BaseOptimizer):
    """Robust Sparsified Distributed Heavy-Ball method.

    Parameters
    ----------
    lr : float
        Global learning rate gamma.  Should satisfy
        gamma <= k / (d * c * L) with c = 23200 and L the smoothness constant
        (see the paper).  Default 0.01.
    momentum : float
        Heavy-Ball momentum coefficient beta, 0 <= beta < 1.  Default 0.8
        (the experimental value in the paper).  Theoretically
        beta = sqrt(1 - 24 * gamma * L).
    compression_ratio : float
        Fraction of dimensions to keep, i.e. k = max(1, round(ratio * d)).
        The compression ratio is alpha = d/k.  Default 0.1 (keep 10%).
    mask_seed : int, optional
        Base seed for the per-round global sparsification mask.  If None, the
        mask is drawn from the global torch RNG (so it respects
        seed_all).  Default None.
    """

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.8,
        compression_ratio: float = 0.1,
        mask_seed: Optional[int] = None,
    ) -> None:
        super().__init__(lr=lr)
        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                f"Heavy-Ball momentum beta must be in [0, 1), got {momentum}."
            )
        if not 0.0 < compression_ratio <= 1.0:
            raise ValueError(
                f"compression_ratio must be in (0, 1], got {compression_ratio}."
            )
        self.momentum = momentum
        self.compression_ratio = compression_ratio
        self.mask_seed = mask_seed

        # --- Lazy-initialised state (set on first gradient) ---
        self.d: Optional[int] = None           # total gradient dimension
        self.k: Optional[int] = None           # number of coordinates to keep
        self.scale: Optional[float] = None     # unbiased scaling factor d/k
        self.reference_shapes: Optional[List[torch.Size]] = None

        # Per-worker server-side momentum buffers: m_i^(t)
        # Indexed by client_id; each is a flat (d,)-shaped tensor.
        # Initialised to the zero vector (m_i^(0) = 0).
        self._momentum_buffers: List[Optional[torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def register_hooks(self) -> None:
        super().register_hooks()
        # Draw the global mask once per round, before any client computes.
        hook_registry.register(HookType.BEFORE_ROUND_SERVER, self._generate_mask)
        # Worker-side: initialise state without changing raw gradients.
        hook_registry.register(HookType.AFTER_COMPUTE, self._client_prepare)
        # Server-side: reconstruct g_tilde_i, update m_i, format for Krum.
        hook_registry.register(HookType.BEFORE_AGGREGATE, self._server_update_momentum)

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _lazy_init(self, sample_grad: List[torch.Tensor]) -> None:
        """Initialise dimensions and buffers on the first gradient computation."""
        self.reference_shapes = [g.shape for g in sample_grad]
        flat = self._flatten(sample_grad)
        self.d = flat.numel()
        # k = round(ratio * d), clamped to [1, d].  Using round (rather than
        # ceil) matches the "keep k coordinates" phrasing of the algorithm.
        self.k = max(1, min(self.d, int(round(self.compression_ratio * self.d))))
        self.scale = self.d / float(self.k)

        print(
            f"[RoSDHB] Initialised: d={self.d}, k={self.k}, "
            f"compression_ratio={self.compression_ratio} "
            f"(alpha=d/k={self.d}/{self.k}={self.scale:.2f}x), "
            f"beta={self.momentum}, gamma={self.lr}"
        )
        print(
            f"[RoSDHB] Random mask keeps {self.k}/{self.d} coordinates "
            f"({self.k / self.d * 100:.1f}%), unbiased scale d/k={self.scale:.4f}"
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
    # Hook 1 — BEFORE_ROUND_SERVER: generate the global sparsification mask
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_mask(self, context: Context) -> None:
        """Draw the global random mask {l_1, ..., l_k} for this round.

        The mask is drawn uniformly without replacement from {0, ..., d-1} and
        stored in context.extra so the BEFORE_AGGREGATE hook uses the same
        mask for every client message in that round. This mirrors the paper's
        "shared PRNG seed" mechanism without changing the framework API.

        The mask is regenerated every round (fresh randomness per round), as
        required for the unbiasedness/variance bound of the random-k
        sparsifier.
        """
        # On the very first round we may not know d yet (no gradient computed).
        # In that case defer: the mask will be created lazily inside
        # _client_prepare on the first honest gradient, and the server-side
        # hook will then read it from context.extra.
        if self.d is None:
            return

        # Fresh mask each round.
        if self.mask_seed is not None:
            # Deterministic per-round seed derived from the base seed + round,
            # so server and a hypothetical separate worker process using the
            # same scheme would agree.
            gen = torch.Generator()
            gen.manual_seed(int(self.mask_seed) + int(context.current_round))
            perm = torch.randperm(self.d, generator=gen)
        else:
            perm = torch.randperm(self.d)
        mask_indices = perm[: self.k].to(torch.long)
        context.extra["rosdhb_mask_indices"] = mask_indices

    # ------------------------------------------------------------------
    # Hook 2 — AFTER_COMPUTE: lazy init and mask preparation only
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _client_prepare(self, context: Context) -> None:
        """Initialise optimizer state without modifying the computed gradient.

        The coordinator snapshots ``context.all_honest_gradients`` after all
        honest clients complete their ``AFTER_COMPUTE`` hooks.  Leaving
        ``context.grad`` untouched here ensures Byzantine attacks compute from
        raw honest backward gradients, not compressed messages.
        """
        client_id = context.current_client_id
        client_grad = context.grad[client_id]

        # --- Lazy init on first gradient ---
        if self.d is None and client_grad is not None:
            self._lazy_init(client_grad)
            # Now that d/k are known, generate this round's mask if it hasn't
            # been (covers the very first round where _generate_mask deferred).
            self._ensure_mask(context)

    def _ensure_mask(self, context: Context) -> torch.Tensor:
        """Return this round's global mask, creating it lazily if needed."""
        mask_indices = context.extra.get("rosdhb_mask_indices")
        if mask_indices is not None:
            return mask_indices

        if self.mask_seed is not None:
            gen = torch.Generator()
            gen.manual_seed(int(self.mask_seed) + int(context.current_round))
            perm = torch.randperm(self.d, generator=gen)
        else:
            perm = torch.randperm(self.d)
        mask_indices = perm[: self.k].to(torch.long)
        context.extra["rosdhb_mask_indices"] = mask_indices
        return mask_indices

    def _compress_full_vector(self, flat_grad: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
        """Apply RoSDHB's random-k upload compression to a full d-vector."""
        mask_indices = mask_indices.to(flat_grad.device)
        return flat_grad[mask_indices]

    def _reconstruct_compressed(
        self,
        compressed: torch.Tensor,
        mask_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Unbiased reconstruction from a length-k compressed message."""
        g_tilde = torch.zeros(self.d, device=device, dtype=compressed.dtype)
        idx = mask_indices.to(device)
        g_tilde[idx] = compressed.to(device) * self.scale
        return g_tilde

    # ------------------------------------------------------------------
    # Hook 3 — BEFORE_AGGREGATE: reconstruct g_tilde_i, update m_i
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _server_update_momentum(self, context: Context) -> None:
        """Server reconstructs g_tilde_i and updates the per-worker momentum.

        For every worker i (honest + byzantine), context.grad[i] holds the
        raw message produced this round:
          - Honest worker: a per-layer raw gradient list.
          - Byzantine worker: usually a forged per-layer full gradient from
            an attack that read raw honest gradients.

        The server:
          1. Applies the global mask compression to full-gradient messages,
             then reconstructs g_tilde_i by placing the k values at the masked
             positions and scaling by d/k for unbiasedness.
          2. Also accepts an already-compressed singleton length-k message and
             reconstructs it directly for compatibility with custom attacks.
          3. Updates the server-side momentum:
                m_i <- beta * m_i + (1 - beta) * g_tilde_i
             with m_i^(0) = 0.
          4. Replaces context.grad[i] with [m_i] so the Krum aggregator
             operates on the momentum set {m_i}_{i=1..n}.
        """
        device = self._infer_device(context)
        if self.d is None:
            for g in context.grad:
                if g is not None:
                    self._lazy_init(g)
                    break
        mask_indices = self._ensure_mask(context)

        for i in range(len(context.grad)):
            g = context.grad[i]
            if g is None:
                # A worker that stayed silent this round (e.g. a sampling-based
                # variant).  Reuse its cached momentum m_i so the aggregator
                # still sees a full n-length list; if no cache yet, use 0.
                while len(self._momentum_buffers) <= i:
                    self._momentum_buffers.append(None)
                m_i = self._momentum_buffers[i]
                if m_i is None:
                    m_i = torch.zeros(self.d, device=device)
                else:
                    m_i = m_i.to(device)
                context.grad[i] = [m_i]
                continue

            # Ensure the momentum buffer list is large enough.
            while len(self._momentum_buffers) <= i:
                self._momentum_buffers.append(None)

            # --- Reconstruct g_tilde_i ---
            if (
                len(g) == 1
                and g[0].dim() == 1
                and g[0].numel() == self.k
            ):
                # Already-compressed forged message: reconstruct directly.
                compressed = g[0].to(device)
                g_tilde = self._reconstruct_compressed(compressed, mask_indices, device)
            elif (
                len(g) == 1
                and g[0].dim() == 1
                and g[0].numel() == self.d
            ):
                # Full flat message: compress after attack, then reconstruct.
                flat = g[0].to(device)
                compressed = self._compress_full_vector(flat, mask_indices)
                g_tilde = self._reconstruct_compressed(compressed, mask_indices, device)
            else:
                # Per-layer raw gradient list: compress after attack, then reconstruct.
                flat = self._flatten(g).to(device)
                compressed = self._compress_full_vector(flat, mask_indices)
                g_tilde = self._reconstruct_compressed(compressed, mask_indices, device)

            # --- Server-side Heavy-Ball momentum update ---
            # m_i <- beta * m_i + (1 - beta) * g_tilde_i,  m_i^(0) = 0.
            m_old = self._momentum_buffers[i]
            if m_old is None:
                m_new = (1.0 - self.momentum) * g_tilde
            else:
                m_old = m_old.to(device)
                m_new = self.momentum * m_old + (1.0 - self.momentum) * g_tilde
            self._momentum_buffers[i] = m_new

            # Replace the uploaded message with the momentum vector.
            context.grad[i] = [m_new]

    # ------------------------------------------------------------------
    # Server-side Heavy-Ball model update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, server, aggregated_grad) -> None:
        """Apply the robustly-aggregated momentum: theta <- theta - gamma * R.

        Parameters
        ----------
        server : Server
            The central server holding the global model.
        aggregated_grad : List[Tensor]
            A singleton list containing the (d,)-shaped flat vector produced
            by the robust aggregator (Krum selects one worker's m_i, which is
            exactly R = F(m_1, ..., m_n)).
        """
        if server is None or aggregated_grad is None:
            print(
                "[RoSDHB] Warning: server or aggregated_grad is None, "
                "skipping update."
            )
            return

        # Krum returns [flat_vector] — the selected worker's m_i.
        g_aggregated = aggregated_grad[0]  # shape (d,)

        device = next(server.model.parameters()).device
        g_aggregated = g_aggregated.to(device, non_blocking=True)

        # Reshape to per-parameter tensors and apply Heavy-Ball update:
        # theta <- theta - gamma * R
        updates = self._unflatten(g_aggregated)

        for param, update in zip(server.model.parameters(), updates):
            param.data.add_(update.to(param.device), alpha=-self.lr)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_device(self, context: Context) -> torch.device:
        """Best-effort device inference for buffers."""
        for g in context.grad:
            if g is not None:
                try:
                    return g[0].device
                except (IndexError, AttributeError):
                    pass
        if context.server is not None:
            try:
                return next(context.server.model.parameters()).device
            except (StopIteration, AttributeError):
                pass
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # State persistence (checkpoint / resume)
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return serialisable state for checkpointing."""
        state: dict = {
            "lr": self.lr,
            "momentum": self.momentum,
            "compression_ratio": self.compression_ratio,
            "mask_seed": self.mask_seed,
            "d": self.d,
            "k": self.k,
            "scale": self.scale,
            "reference_shapes": self.reference_shapes,
        }

        # Momentum buffers — move to CPU for serialisation.
        if self._momentum_buffers:
            state["momentum_buffers"] = [
                buf.cpu().clone() if buf is not None else None
                for buf in self._momentum_buffers
            ]

        return state

    def set_state(self, state: dict) -> None:
        """Restore state from a checkpoint."""
        self.lr = state.get("lr", self.lr)
        self.momentum = state.get("momentum", self.momentum)
        self.compression_ratio = state.get("compression_ratio", self.compression_ratio)
        self.mask_seed = state.get("mask_seed", self.mask_seed)
        self.d = state.get("d")
        self.k = state.get("k")
        self.scale = state.get("scale")
        self.reference_shapes = state.get("reference_shapes")

        raw_mom = state.get("momentum_buffers")
        if raw_mom is not None:
            self._momentum_buffers = [
                buf.clone() if buf is not None else None for buf in raw_mom
            ]
