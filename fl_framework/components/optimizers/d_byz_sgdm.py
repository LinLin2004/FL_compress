# fl_framework/components/optimizers/d_byz_sgdm.py
"""D-Byz-SGDM: Delayed Momentum Aggregation for Byzantine-robust FL.

Combines:
  - Independent Bernoulli client sampling (each client is selected with
    probability p every step)
  - Client-side stochastic gradient descent with momentum (SGDM)
  - Server-side *delayed* momentum aggregation: the server keeps a momentum
    cache m_i for every client i (honest + byzantine).  A client that is
    selected this step contributes a freshly updated momentum
    m'_i = (1-alpha)*m_i + alpha*g; a client that is *not* selected
    contributes its stale cached momentum m'_i = m_i (implicit delay
    tau(i,t), no explicit bookkeeping needed).
  - A (delta, c)-robust aggregator (here the framework's KrumAggregator)
    applied to the full list of n momentum vectors {m'_i} — both fresh and
    stale — so the aggregator sees every client every step.

Algorithm outline (per step t):

  Sampling (server-side, BEFORE_STEP_SERVER):
    For each client i = 1..n:
        draw Bernoulli(p) -> S_t (the set of participants this step)

  Honest client i in S_t (AFTER_COMPUTE):
    1. Sample a mini-batch from D_i
    2. Compute stochastic gradient  g = grad F_i(x; xi)
    3. Update momentum:  m'_i = (1-alpha)*m_i + alpha*g   (EMA)
    4. DEFER writing m'_i to context.grad — store in pending buffer
       so that context.all_honest_gradients still contains the RAW
       per-layer gradients when Byzantine attacks are computed.
       The pending buffer is flushed when the first Byzantine client
       is processed, or in BEFORE_AGGREGATE if num_byzantine=0.

  Honest client i NOT in S_t:
    - context.grad[i] remains None; the server fills it with the cached
      momentum m'_i = m_i in BEFORE_AGGREGATE.  No local computation.

  Byzantine client i in S_t (AFTER_COMPUTE):
    - Runs its attack (which reads context.all_honest_gradients containing
      the RAW honest gradients), producing an arbitrary forged vector.
    - The forged vector is flattened to [flat_vector] format and treated
      as this client's m'_i.
    - All pending honest momentum results are flushed into context.grad
      at this point (safe because the attack has already read the raw
      gradients).

  Byzantine client i NOT in S_t:
    - context.grad[i] is set to None; the server fills it with the cached
      forged value m'_i = m_i (the last value this byzantine client sent
      when it was last selected).  Same delayed-momentum treatment as
      honest clients.

  Server (BEFORE_AGGREGATE):
    5. For every client i with context.grad[i] is None (i.e. any client
       not selected this step), set context.grad[i] = [m_i] using the
       cached momentum.  Now all n entries are populated.
    6. Assemble M = {m'_i}_{i=1..n} and let agg = Agg(M) (Krum).

  Server (step / BEFORE_UPDATE):
    7. Update global model:  x <- x - eta * agg
    8. Refresh the momentum cache for every client:
       m_i <- m'_i  (the freshly updated value for selected clients, the
       unchanged stale value for non-selected clients — both already live
       in context.grad[i] after step 5, so the cache simply adopts them).

Symbol glossary:
  n       - total number of clients (honest + byzantine)
  G       - set of honest clients, |G| > n/2  (byzantine ratio delta < 0.5)
  p       - per-step per-client selection probability, 0 < p <= 1
  alpha   - client momentum coefficient (EMA weight on the new gradient),
            e.g. 0.9
  eta     - global learning rate
  Agg     - (delta, c)-robust aggregator (KrumAggregator in this framework)
  m_i     - server-cached momentum for client i (d-vector)
  m'_i    - this step's momentum for client i
  g       - stochastic gradient at the current global model x
  S_t     - set of clients selected this step

Implementation notes (avoiding ambiguity):
  * Momentum cache lives on the server (this optimizer instance).  Memory
    footprint matches the full-participation setting; no extra
    communication because non-selected clients simply stay silent and the
    server reuses their cached momentum.
  * The implicit delay tau(i,t) is handled by reusing the cache — no
    explicit delay counter is maintained.
  * Independent Bernoulli sampling is used here; the same idea generalises
    to other sampling schemes by swapping the sampler in _sample_clients.

Hooks used:
  - BEFORE_STEP_SERVER : sample S_t for this step
  - AFTER_COMPUTE      : selected clients do the momentum update (honest
                         clients: EMA, DEFERRED via pending buffer so that
                         Byzantine attacks see raw gradients; byzantine
                         clients: flatten attack output, flush pending);
                         non-selected clients are marked silent (grad=None)
  - BEFORE_AGGREGATE   : flush any remaining pending momentum results
                         (when num_byzantine=0); fill non-selected clients
                         (honest & byzantine) with cached momentum
                         so Agg sees all n vectors
  - AFTER_AGGREGATE    : refresh the momentum cache m_i <- m'_i
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .base_optimizer import BaseOptimizer
from fl_framework.core.hooks import HookType, Context, hook_registry


class DByzSGDM(BaseOptimizer):
    """D-Byz-SGDM: Delayed Momentum Aggregation with robust aggregation.

    Parameters
    ----------
    lr : float
        Global learning rate eta.  Default 0.1.
    momentum : float
        Client momentum coefficient alpha (EMA weight on the new gradient):
        m'_i = (1-alpha)*m_i + alpha*g.  Default 0.9.
    participation : float
        Per-step per-client selection probability p, 0 < p <= 1.  Default 1.0
        (full participation, equivalent to vanilla Byz-SGDM with momentum).
    weight_decay : float
        L2 regularization factor applied to the local gradient before the
        momentum update.  Default 0.0.
    """

    def __init__(
        self,
        lr: float = 0.1,
        momentum: float = 0.9,
        participation: float = 1.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(lr=lr)
        self.momentum = momentum
        self.participation = participation
        self.weight_decay = weight_decay

        # --- Lazy-initialised state (set on first gradient) ---
        self.d: Optional[int] = None  # total gradient dimension
        self.reference_shapes: Optional[List[torch.Size]] = None

        # Server-side per-client momentum cache: m_i
        # Indexed by client_id; each is a flat (d,)-shaped tensor stored on CPU.
        # This cache is the heart of the "delayed momentum" idea: non-selected
        # clients contribute their cached m_i instead of computing a fresh one.
        self._momentum_cache: List[Optional[torch.Tensor]] = []

        # Pending momentum results for honest clients.
        # Keyed by client_id; each value is a singleton list [flat_tensor]
        # containing the momentum-updated vector m'_i.
        # These are computed during honest clients' AFTER_COMPUTE hook but
        # NOT written to context.grad immediately — they are deferred so
        # that context.all_honest_gradients still contains RAW per-layer
        # gradients when Byzantine attacks are computed.  They are flushed
        # into context.grad when the first Byzantine client is processed,
        # or in BEFORE_AGGREGATE when num_byzantine=0.
        self._pending_momentum: dict = {}

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def register_hooks(self) -> None:
        super().register_hooks()
        hook_registry.register(HookType.BEFORE_STEP_SERVER, self._sample_clients)
        hook_registry.register(HookType.AFTER_COMPUTE, self._client_process)
        hook_registry.register(HookType.BEFORE_AGGREGATE, self._fill_non_participants)
        hook_registry.register(HookType.AFTER_AGGREGATE, self._refresh_cache)

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _lazy_init(self, sample_grad: List[torch.Tensor]) -> None:
        """Initialise dimensions on the first gradient computation."""
        self.reference_shapes = [g.shape for g in sample_grad]
        flat = self._flatten(sample_grad)
        self.d = flat.numel()
        print(
            f"[D-Byz-SGDM] Initialised: d={self.d}, participation p={self.participation}, "
            f"momentum alpha={self.momentum}, lr={self.lr}, weight_decay={self.weight_decay}"
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
    # Hook 1 — BEFORE_STEP_SERVER: independent Bernoulli sampling of S_t
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_clients(self, context: Context) -> None:
        """Draw the participant set S_t for this step.

        Each client is selected independently with probability p.  The
        resulting boolean mask is stored in context.extra so that the
        AFTER_COMPUTE / BEFORE_AGGREGATE hooks can consult it.  A fresh
        draw is made every step.
        """
        n = len(context.clients)
        if self.participation >= 1.0:
            # Full participation: everyone participates, no randomness.
            mask = [True] * n
        else:
            mask = [
                bool(torch.rand(1).item() < self.participation) for _ in range(n)
            ]
        context.extra["d_byz_sgdm_participants"] = mask

    # ------------------------------------------------------------------
    # Hook 2 — AFTER_COMPUTE: client-side momentum update (participants only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _client_process(self, context: Context) -> None:
        """Per-client momentum EMA for *selected honest* clients.

        Honest clients:
          Compute the momentum EMA update m'_i = (1-alpha)*m_i + alpha*g,
          but DEFER writing the result to context.grad.  Instead, store it
          in self._pending_momentum.  This ensures that
          context.all_honest_gradients (set by the coordinator after all
          honest clients finish) still contains the RAW per-layer gradients,
          so Byzantine attacks can compute on the original gradients.

        Byzantine clients:
          Flatten the forged gradient into singleton [flat_vector] format,
          then flush all pending honest momentum results into context.grad.
          This guarantees that:
          1. Byzantine attacks saw raw honest gradients
          2. All context.grad entries are in the correct format after this
             hook returns

        Non-selected honest clients are skipped: context.grad[client_id]
        stays None and is filled later from the cache in _fill_non_participants.
        """
        client_id = context.current_client_id
        client = context.clients[client_id]
        client_grad = context.grad[client_id]

        # --- Lazy init on first call ---
        if self.d is None and client_grad is not None:
            self._lazy_init(client_grad)

        # --- Ensure cache list is large enough ---
        while len(self._momentum_cache) <= client_id:
            self._momentum_cache.append(None)

        # --- Byzantine clients: flatten forged gradient, then flush pending ---
        if client.client_type == "Byzantine":
            mask = context.extra.get("d_byz_sgdm_participants", None)
            if mask is not None and not mask[client_id]:
                # Non-sampled Byzantine: silent, reuse cached momentum later.
                context.grad[client_id] = None
            else:
                # Sampled Byzantine: flatten the forged gradient into
                # singleton [flat_vector] format.
                if client_grad is not None and not (
                    len(client_grad) == 1
                    and client_grad[0].dim() == 1
                    and client_grad[0].numel() == self.d
                ):
                    context.grad[client_id] = [self._flatten(client_grad)]

            # Flush all pending honest momentum results into context.grad.
            # This is the key step: byzantine attacks have already been
            # computed using the RAW honest gradients in
            # context.all_honest_gradients, so it is now safe to overwrite
            # the honest clients' context.grad entries with the momentum
            # results.
            for hid, hmom in self._pending_momentum.items():
                context.grad[hid] = hmom
            self._pending_momentum.clear()
            return

        # --- Look up whether this honest client was selected this step ---
        mask = context.extra.get("d_byz_sgdm_participants", None)
        if mask is not None and not mask[client_id]:
            # Non-selected honest client: produce nothing here.  The server
            # will reuse the cached momentum m_i in _fill_non_participants.
            # Drop the freshly computed gradient (the client stayed "silent").
            context.grad[client_id] = None
            return

        # --- Selected honest client: EMA momentum update ---
        device = client_grad[0].device
        flat_grad = self._flatten(client_grad).to(device)  # g

        # Optional L2 weight decay on the gradient
        if self.weight_decay > 0:
            model = client.model
            param_flat = self._flatten([p.data for p in model.parameters()]).to(device)
            flat_grad.add_(param_flat, alpha=self.weight_decay)

        # m'_i = (1-alpha)*m_i + alpha*g   (first iteration: m_i = 0)
        m_old = self._momentum_cache[client_id]
        if m_old is None:
            m_new = flat_grad.clone()
        else:
            m_old = m_old.to(device)
            m_new = (1.0 - self.momentum) * m_old + self.momentum * flat_grad

        # DEFER: do NOT write to context.grad yet — store in pending buffer
        # so that context.all_honest_gradients still contains raw gradients
        # when Byzantine attacks are computed later.
        self._pending_momentum[client_id] = [m_new]

    # ------------------------------------------------------------------
    # Hook 3 — BEFORE_AGGREGATE: fill non-participants with cached momentum
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _fill_non_participants(self, context: Context) -> None:
        """Ensure all n entries of context.grad are populated as [flat (d,)-tensor].

        First, flush any pending honest momentum results into context.grad.
        This handles the case where num_byzantine=0 (no Byzantine clients
        to trigger the flush in _client_process).

        After the AFTER_COMPUTE pass:
          - Selected honest clients:    context.grad[i] = [m'_i] (fresh, flat)
          - Byzantine clients:          context.grad[i] = forged m'_i (per-layer list)
          - Non-selected honest clients: context.grad[i] = None  (silent)

        This hook:
          1. Replaces every None entry with the client's cached momentum
             [m_i], so the aggregator receives the full list of n momentum
             vectors (a mix of fresh and stale) — the "delayed momentum
             aggregation".
          2. Flattens Byzantine clients' per-layer forged gradient into a
             singleton [flat_vector] so that all entries have a uniform
             format.  This is necessary because the Krum aggregator returns
             the winning client's gradient list as-is; if all inputs are
             [flat_vector], the output is also [flat_vector], and step()
             can safely assume aggregated_grad[0] is a flat (d,)-tensor.
        """
        # Flush any pending honest momentum results (needed when num_byzantine=0)
        if self._pending_momentum:
            for hid, hmom in self._pending_momentum.items():
                context.grad[hid] = hmom
            self._pending_momentum.clear()

        for i in range(len(context.grad)):
            g = context.grad[i]

            if g is None:
                # Non-selected client (honest or byzantine): reuse cached
                # momentum m_i.  For honest clients this is the stale EMA
                # momentum from their most recent participation.  For
                # byzantine clients this is the stale forged value from
                # their most recent participation.
                while len(self._momentum_cache) <= i:
                    self._momentum_cache.append(None)

                m_cached = self._momentum_cache[i]
                if m_cached is None:
                    # First time we ever see this client and it was not selected:
                    # there is no cached momentum yet.  Use a zero vector (m_i = 0
                    # is the initial condition), so the aggregator still sees a
                    # valid n-length list.
                    device = self._infer_device(context)
                    m_cached = torch.zeros(self.d, device=device)
                else:
                    device = self._infer_device(context)
                    m_cached = m_cached.to(device)

                context.grad[i] = [m_cached]
            else:
                # Entry is populated.  Ensure it is in [flat (d,)-tensor] format.
                # Byzantine clients may have per-layer lists; flatten them.
                # Selected honest clients already have [flat_vector] format.
                if not (len(g) == 1 and g[0].dim() == 1 and g[0].numel() == self.d):
                    context.grad[i] = [self._flatten(g)]

    # ------------------------------------------------------------------
    # Hook 4 — AFTER_AGGREGATE: refresh the momentum cache m_i <- m'_i
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _refresh_cache(self, context: Context) -> None:
        """Adopt this step's momentum vectors into the cache.

        After _fill_non_participants, context.grad[i] holds m'_i for every
        client i (fresh for selected honest clients, forged for byzantine
        clients, stale-but-valid for non-selected honest clients).  We simply
        store these back into the cache so that the *next* step's
        non-participants reuse the most recent value.
        """
        for i in range(len(context.grad)):
            g = context.grad[i]
            if g is None:
                continue
            # g is a singleton list [flat (d,)-tensor] (or, for byzantine
            # clients, a per-layer list that we flatten to (d,)).
            if len(g) == 1 and g[0].dim() == 1 and g[0].numel() == self.d:
                m_i = g[0]
            else:
                m_i = self._flatten(g)
            self._momentum_cache[i] = m_i.cpu().clone()

    # ------------------------------------------------------------------
    # Server-side model update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, server, aggregated_grad) -> None:
        """Apply the aggregated momentum to update the global model.

            x <- x - eta * agg

        Parameters
        ----------
        server : Server
            The central server holding the global model.
        aggregated_grad : List[Tensor]
            A singleton list containing the (d,)-shaped flat vector produced
            by the robust aggregator (Krum selects one client's m'_i).
        """
        if server is None or aggregated_grad is None:
            print(
                "[D-Byz-SGDM] Warning: server or aggregated_grad is None, "
                "skipping update."
            )
            return

        # Krum returns [flat_vector] — the selected client's m'_i.
        g_aggregated = aggregated_grad[0]  # shape (d,)

        device = next(server.model.parameters()).device
        g_aggregated = g_aggregated.to(device, non_blocking=True)

        # Reshape to per-parameter tensors and apply: x <- x - eta * agg
        updates = self._unflatten(g_aggregated)

        for param, update in zip(server.model.parameters(), updates):
            param.data.add_(update.to(param.device), alpha=-self.lr)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_device(self, context: Context) -> torch.device:
        """Best-effort device inference for cache tensors."""
        # Prefer a populated gradient entry's device.
        for g in context.grad:
            if g is not None:
                try:
                    return g[0].device
                except (IndexError, AttributeError):
                    pass
        # Fall back to the server model's device.
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
            "participation": self.participation,
            "weight_decay": self.weight_decay,
            "d": self.d,
            "reference_shapes": self.reference_shapes,
        }

        # Momentum cache — move to CPU for serialisation.
        if self._momentum_cache:
            state["momentum_cache"] = [
                buf.cpu().clone() if buf is not None else None
                for buf in self._momentum_cache
            ]

        return state

    def set_state(self, state: dict) -> None:
        """Restore state from a checkpoint."""
        self.lr = state.get("lr", self.lr)
        self.momentum = state.get("momentum", self.momentum)
        self.participation = state.get("participation", self.participation)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        self.d = state.get("d")
        self.reference_shapes = state.get("reference_shapes")

        raw_cache = state.get("momentum_cache")
        if raw_cache is not None:
            self._momentum_cache = [
                buf.clone() if buf is not None else None for buf in raw_cache
            ]
