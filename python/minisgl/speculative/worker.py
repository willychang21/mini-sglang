"""Speculative decoding for a *batch* of running requests (continuous batching).

``SpeculativeDecoder`` (decoder.py) runs one request on a private, contiguous
slot range. Here the same draft -> verify -> commit round runs for every
decoding request at once, on the scheduler's paged KV cache, its radix cache
and its admission / retraction machinery. The mapping to SGLang:

=================================  ==========================================
mini-sglang                        SGLang
=================================  ==========================================
``SpecWorker``                     ``NGRAMWorker`` / ``StandaloneWorker``
                                   (``forward_batch_generation``)
``SpecVerifyInfo``                 ``batch.spec_info`` (``NgramVerifyInput``,
                                   ``EagleVerifyInput``)
``Batch(phase="verify")``          ``ForwardMode.TARGET_VERIFY``
``DraftModelDrafter`` catch-up     ``ForwardMode.DRAFT_EXTEND``
``SpecWorker.commit``              ``verify()`` + ``eagle_sample`` + freeing
                                   the rejected slots
=================================  ==========================================

Invariants (``L`` = confirmed tokens of a request, i.e. ``len(input_ids)``).

Between rounds (exactly like plain decode)::

    cached_len == L - 1     K/V valid for [0, L-1); the newest token has none
    device_len == L
    pages cover [0, ceil(cached_len / page_size) * page_size)

``prepare`` (one round)::

    plan k_i <= min(k, remaining - 1)       never write past max_device_len - 1
    allocate pages for [cached_len, L + k_i)
    draft d1..dk_i (a draft model writes its own K/V into the *same slots*
                    of its own pool, positions [draft_len, L + k_i - 1))
    token_pool[row, L : L + k_i] = d1..dk_i
    device_len = L + k_i  -> the verify batch extends [L-1, L+k_i)

``commit`` (after the target forward wrote K/V for [L-1, L+k_i))::

    committed = accepted proposals + correction/bonus   (m = len(committed))
    cached_len = L + m - 1, device_len = L + m      (back to the invariant)
    K/V valid on the confirmed path: [0, L + n_accepted) == [0, cached_len)
    free pages past ceil(cached_len): they held only rejected proposals

So the radix cache (``cache_req``), retraction and aborts - which all run
*between* rounds - see the same state as after a plain decode step.
"""

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Collection, List, Tuple

import torch

from ..core import Batch, Req, SamplingParams
from ..utils import div_ceil, init_logger
from .drafter import NgramDrafter
from .verify import greedy_commit_ids, logits_to_probs, rejection_commit

if TYPE_CHECKING:
    from ..engine import Engine
    from ..scheduler.cache import CacheManager
    from ..scheduler.config import SchedulerConfig

logger = init_logger(__name__)

Proposals = Tuple[List[List[int]], List[torch.Tensor | None]]


@dataclass
class SpecVerifyInfo:
    """What a verify batch carries besides its requests (SGLang ``spec_info``)."""

    seq_lens: List[int]  # L: confirmed tokens of each request before the round
    proposals: List[List[int]]  # d1..dk_i (k_i may be 0: plain decode for that request)
    draft_probs: List[torch.Tensor | None]  # [k_i, V] draft distributions q (sampling only)
    alloc_lens: List[int]  # pages were allocated for [0, alloc_len)

    @property
    def num_rows(self) -> List[int]:
        """Verify rows per request: ``[last_token, d1..dk_i]``."""
        return [len(p) + 1 for p in self.proposals]


@dataclass
class SpecRoundStats:
    verify_steps: int = 0  # target forwards in verify mode
    req_rounds: int = 0  # (request, round) pairs
    proposed: int = 0
    accepted: int = 0
    committed: int = 0  # tokens produced by verify steps (accepted + correction/bonus)
    trimmed: int = 0  # proposals dropped because the round's pages did not fit
    accepted_per_round: List[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(self.proposed, 1)

    @property
    def accept_length(self) -> float:
        """Tokens committed per request per verify step (SGLang's
        ``spec_accept_length``; 1.0 means speculation gained nothing)."""
        return self.committed / max(self.req_rounds, 1)


def _eos_for(sp: SamplingParams, eos_ids: Collection[int]) -> Collection[int]:
    return () if sp.ignore_eos else eos_ids


# --------------------------------------------------------------------- drafters
class BatchDrafter(ABC):
    # True: proposing writes K/V into the request's pages, so the round's pages
    # must be allocated first. False: proposals are known before allocation,
    # so we only allocate for what was actually proposed.
    needs_kv: bool = False
    forwards: int = 0  # draft model forward passes

    @abstractmethod
    def propose(
        self,
        reqs: List[Req],
        seq_lens: List[int],
        max_ks: List[int],
        token_pool: torch.Tensor,
        gen: torch.Generator,
    ) -> Proposals: ...

    def on_commit(self, req: Req, seq_len: int, num_proposed: int, num_accepted: int) -> None:
        """Called once per request after verification (rollback hook)."""

    def shutdown(self) -> None:
        pass


class NgramBatchDrafter(BatchDrafter):
    """Prompt lookup per request (SGLang ``NGRAMWorker`` keeps a trie over
    many requests' tokens instead; same idea, no model)."""

    def __init__(self, vocab_size: int, eos_ids: Collection[int], max_ngram: int = 4) -> None:
        self.inner = NgramDrafter(vocab_size, max_ngram=max_ngram)
        self.eos_ids = eos_ids

    def propose(self, reqs, seq_lens, max_ks, token_pool, gen) -> Proposals:
        proposals: List[List[int]] = []
        probs: List[torch.Tensor | None] = []
        for req, k in zip(reqs, max_ks):
            if k <= 0:
                proposals.append([])
                probs.append(None)
                continue
            sp = req.sampling_params
            seq = req.input_ids.tolist()
            props, q = self.inner.propose(seq, k, sp, _eos_for(sp, self.eos_ids), gen)
            proposals.append(list(props))
            probs.append(torch.stack(q) if q else None)
        return proposals, probs


@dataclass(eq=False)
class _DraftSeq:
    """A request as the draft engine sees it: same row, its own lengths."""

    table_idx: int
    cached_len: int
    device_len: int

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1


class DraftModelDrafter(BatchDrafter):
    """A smaller model drafts autoregressively for the whole batch.

    Its KV pool is separate but indexed by the *target's* page table, so a
    token's draft K/V lives at the same slot number as its target K/V and the
    target's allocator, radix cache, retraction and abort logic manage both
    (SGLang's EAGLE / standalone workers share ``req_to_token_pool`` and the
    allocator the same way).

    ``draft_len[req]``: positions ``[0, draft_len)`` hold valid draft K/V. It
    starts at 0 for every admitted request: the first round's catch-up
    ("draft extend") runs the draft over the whole prompt. Each round then
    catches up ``[draft_len, L)``, proposes, and after commit rolls back to
    the confirmed path.
    """

    needs_kv = True

    def __init__(self, engine: "Engine", eos_ids: Collection[int]) -> None:
        self.engine = engine
        self.eos_ids = eos_ids
        self.vocab_size = engine.model_config.vocab_size
        self.forwards = 0
        self.draft_lens: "weakref.WeakKeyDictionary[Req, int]" = weakref.WeakKeyDictionary()

    def _forward(self, seqs: List[_DraftSeq], phase: str, token_pool: torch.Tensor) -> torch.Tensor:
        engine = self.engine
        device = engine.device
        rows = torch.cat([torch.full((s.extend_len,), s.table_idx, dtype=torch.int64) for s in seqs])
        pos = torch.cat([torch.arange(s.cached_len, s.device_len) for s in seqs])
        mapping = (rows.to(device), pos.to(device))
        batch = Batch(reqs=seqs, phase=phase)  # type: ignore[arg-type]
        batch.input_ids = token_pool[mapping]
        batch.positions = mapping[1]
        batch.out_loc = engine.page_table[mapping]
        engine.attn_backend.prepare_metadata(batch)
        self.forwards += 1
        return engine.forward_batch(batch, None).logits  # [len(seqs), V]; advances seqs

    def _pick(self, logits: torch.Tensor, reqs: List[Req], gen: torch.Generator):
        greedy = logits.argmax(dim=-1).tolist()
        out = []
        for i, req in enumerate(reqs):
            sp = req.sampling_params
            if sp.is_greedy:
                out.append((int(greedy[i]), None))
            else:
                q = logits_to_probs(logits[i].float().cpu(), sp.temperature, sp.top_k, sp.top_p)
                out.append((int(torch.multinomial(q, 1, generator=gen)), q))
        return out

    def propose(self, reqs, seq_lens, max_ks, token_pool, gen) -> Proposals:
        n = len(reqs)
        proposals: List[List[int]] = [[] for _ in range(n)]
        qs: List[List[torch.Tensor]] = [[] for _ in range(n)]
        active = [i for i in range(n) if max_ks[i] > 0]
        # draft extend: catch up on [draft_len, L), the last row predicts d1
        seqs = {
            i: _DraftSeq(reqs[i].table_idx, self.draft_lens.get(reqs[i], 0), seq_lens[i]) for i in active
        }
        phase = "prefill"
        while active:
            logits = self._forward([seqs[i] for i in active], phase, token_pool)
            nxt = []
            for i, (tok, q) in zip(active, self._pick(logits, [reqs[i] for i in active], gen)):
                token_pool[reqs[i].table_idx, seq_lens[i] + len(proposals[i])] = tok
                proposals[i].append(tok)
                if q is not None:
                    qs[i].append(q)
                eos = _eos_for(reqs[i].sampling_params, self.eos_ids)
                if len(proposals[i]) < max_ks[i] and tok not in eos:
                    nxt.append(i)
            active, phase = nxt, "decode"  # draft decode: feed d_j, predict d_{j+1}
        probs = [torch.stack(q) if q else None for q in qs]
        return proposals, probs

    def shutdown(self) -> None:
        self.engine.shutdown()

    def on_commit(self, req, seq_len, num_proposed, num_accepted) -> None:
        if num_proposed == 0:
            return  # the draft did not run for this request this round
        # the draft wrote [.., L + k - 1): its last proposal was never fed back
        self.draft_lens[req] = min(seq_len + num_proposed - 1, seq_len + num_accepted, req.cached_len)


# ----------------------------------------------------------------------- worker
class SpecWorker:
    """Plans, drafts and commits one speculative round for a decode batch.

    The scheduler owns the forward pass in between: ``prepare`` -> build a
    ``verify`` batch over ``[cached_len, device_len)`` of every request ->
    ``Engine.forward_batch`` with all logits -> ``commit``.
    """

    def __init__(
        self,
        drafter: BatchDrafter,
        num_draft_tokens: int,
        cache_manager: "CacheManager",
        token_pool: torch.Tensor,
        eos_ids: Collection[int],
        seed: int = 0,
    ) -> None:
        self.drafter = drafter
        self.k = num_draft_tokens
        self.cache_manager = cache_manager
        self.token_pool = token_pool
        self.eos_ids = eos_ids
        # one CPU generator for draft sampling and rejection sampling: every TP
        # rank seeds it identically and sees identical logits
        self.generator = torch.Generator().manual_seed(seed)
        self.stats = SpecRoundStats()

    # --------------------------------------------------------------- prepare
    def _fit_to_pages(self, reqs: List[Req], seq_lens: List[int], ks: List[int]) -> List[int]:
        """Shrink proposals so the round's pages fit. Admission reserves k extra
        tokens per request, so this only bites under optimistic admission (the
        scheduler has already retracted until a plain decode step fits)."""
        cm, ps = self.cache_manager, self.cache_manager.page_size
        spare = cm.available_pages - cm.pages_needed(reqs)  # beyond a plain decode step
        fitted = []
        for L, k in zip(seq_lens, ks):
            base = div_ceil(L, ps)
            while k > 0 and div_ceil(L + k, ps) - base > spare:
                k -= 1
                self.stats.trimmed += 1
            spare -= div_ceil(L + k, ps) - base
            fitted.append(k)
        return fitted

    def prepare(self, reqs: List[Req]) -> SpecVerifyInfo | None:
        """Draft for ``reqs`` and set their lengths for the verify batch.
        Returns None when nobody has a proposal (run a plain decode step)."""
        seq_lens = []
        for req in reqs:
            L = req.device_len
            assert req.cached_len == L - 1 and len(req.input_ids) == L, req
            seq_lens.append(L)
        # k_i <= remaining - 1: the bonus token of a fully accepted round is
        # the last one the budget allows, so K/V is never written at or past
        # max_device_len (<= max_seq_len, the page-table width)
        max_ks = [max(0, min(self.k, r.max_device_len - L - 1)) for r, L in zip(reqs, seq_lens)]
        proposals: List[List[int]] | None = None
        probs: List[torch.Tensor | None] | None = None
        if not self.drafter.needs_kv:
            proposals, probs = self.drafter.propose(reqs, seq_lens, max_ks, self.token_pool, self.generator)
            max_ks = [len(p) for p in proposals]
        ks = self._fit_to_pages(reqs, seq_lens, max_ks)
        if not any(ks):
            return None
        for req, L, k in zip(reqs, seq_lens, ks):
            req.device_len = L + k
        self.cache_manager.allocate_paged(reqs)  # pages for [cached_len, L + k)
        if proposals is None:
            proposals, probs = self.drafter.propose(reqs, seq_lens, ks, self.token_pool, self.generator)
        assert probs is not None
        for i, (req, L, k) in enumerate(zip(reqs, seq_lens, ks)):
            props = proposals[i] = proposals[i][:k]
            if probs[i] is not None:
                probs[i] = probs[i][:k]  # type: ignore[index]
            if props:
                self.token_pool[req.table_idx, L : L + len(props)] = torch.tensor(
                    props, dtype=self.token_pool.dtype, device=self.token_pool.device
                )
            req.device_len = L + len(props)  # a draft may stop early at EOS
        return SpecVerifyInfo(
            seq_lens=seq_lens,
            proposals=proposals,
            draft_probs=probs,
            alloc_lens=[L + k for L, k in zip(seq_lens, ks)],
        )

    # ---------------------------------------------------------------- commit
    def commit(self, info: SpecVerifyInfo, reqs: List[Req], logits: torch.Tensor) -> List[List[int]]:
        """Verify every request's proposals against the target's ``logits``
        (rows packed request by request), roll lengths back to the confirmed
        path and free the pages of rejected proposals. Returns the committed
        tokens of each request (host tokens are appended by the caller)."""
        greedy_ids = logits.argmax(dim=-1).tolist()
        committed_all: List[List[int]] = []
        offset = 0
        st = self.stats
        st.verify_steps += 1
        for req, L, props, q, alloc_len, n_rows in zip(
            reqs, info.seq_lens, info.proposals, info.draft_probs, info.alloc_lens, info.num_rows
        ):
            sp = req.sampling_params
            remaining = req.max_device_len - L
            eos = _eos_for(sp, self.eos_ids)
            if sp.is_greedy:
                committed = greedy_commit_ids(props, greedy_ids[offset : offset + n_rows], remaining, eos)
            else:
                p = logits_to_probs(
                    logits[offset : offset + n_rows].float().cpu(), sp.temperature, sp.top_k, sp.top_p
                )
                if q is None:
                    q = p.new_zeros(0, p.shape[-1])
                committed = rejection_commit(props, q, p, remaining, eos, self.generator)
            offset += n_rows
            m = len(committed)
            assert m >= 1
            n_acc = 0
            while n_acc < min(len(props), m) and committed[n_acc] == props[n_acc]:
                n_acc += 1

            # ---- roll back to the confirmed path
            self.token_pool[req.table_idx, L : L + m] = torch.tensor(
                committed, dtype=self.token_pool.dtype, device=self.token_pool.device
            )
            req.cached_len = L + m - 1  # == L + n_acc (or L + n_acc - 1 if it stopped on a draft)
            req.device_len = L + m
            self.cache_manager.free_spec_tail(req, alloc_len)
            self.drafter.on_commit(req, L, len(props), n_acc)

            st.req_rounds += 1
            st.proposed += len(props)
            st.accepted += n_acc
            st.committed += m
            st.accepted_per_round.append(n_acc)
            committed_all.append(committed)
        assert offset == logits.shape[0]
        return committed_all


def create_spec_worker(
    sched_config: "SchedulerConfig",
    engine: "Engine",
    cache_manager: "CacheManager",
    token_pool: torch.Tensor,
    eos_ids: Collection[int],
    seed: int,
) -> SpecWorker:
    algo = sched_config.speculative_algorithm
    vocab = engine.model_config.vocab_size
    if algo == "ngram":
        drafter: BatchDrafter = NgramBatchDrafter(vocab, eos_ids)
    elif algo == "draft_model":
        drafter = DraftModelDrafter(_create_draft_engine(sched_config, engine), eos_ids)
    else:
        raise ValueError(f"unknown speculative algorithm {algo!r}")
    return SpecWorker(drafter, sched_config.speculative_num_draft_tokens, cache_manager, token_pool, eos_ids, seed)


def _create_draft_engine(sched_config: "SchedulerConfig", target: "Engine") -> "Engine":
    from ..core import set_global_ctx
    from ..engine import Engine

    if target.tp_info.size > 1:
        raise ValueError("speculative_algorithm='draft_model' does not support tp_size > 1")
    cfg = replace(
        target.config,
        model_path=sched_config.speculative_draft_model or target.config.model_path,
        num_layers=sched_config.speculative_draft_num_layers,
    )
    draft = Engine(cfg, page_table=target.page_table)
    set_global_ctx(target.ctx)  # Engine() installs its own context
    if draft.model_config.vocab_size != target.model_config.vocab_size:
        raise ValueError(
            f"target/draft vocabularies differ ({target.model_config.vocab_size} vs "
            f"{draft.model_config.vocab_size})"
        )
    logger.info(
        "draft model: %s (%d layers), sharing the target page table",
        cfg.model_path, draft.model_config.num_layers,
    )
    return draft
