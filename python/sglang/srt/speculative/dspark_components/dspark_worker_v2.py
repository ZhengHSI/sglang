import logging
import os
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import replace
from typing import Optional

import torch

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    compute_position,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_worker_common import (
    build_block_pos_offsets,
    build_draft_tp_worker,
    make_draft_block_spec_info,
    make_draft_sampler_capture_hook,
)
from sglang.srt.speculative.dspark_components.dspark_config import (
    DSV4_DRAFT_ATTENTION_BACKEND,
    draft_is_deepseek_v4,
    resolve_runtime_config,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    DraftBlockResult,
    make_next_draft_input,
    maybe_build_draft_sampler,
    resolve_greedy_mask,
)
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_observability import (
    DsparkStepObservers,
    InfoSegment,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkVerifyPlanner,
    alloc_verify_window,
    dp_global_verify_tier_num_tokens,
    idle_ragged_layout,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    CommitInjectCtx,
    DSparkPPVerifyInputRaw,
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
    verify_logits_adjustments_are_noop,
)
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.srt.utils import get_available_gpu_memory, is_cuda

logger = logging.getLogger(__name__)


class _PPDraftLogitsCacheEntry:
    __slots__ = ("corrected_logits", "draft_tokens", "num_rows")

    def __init__(
        self,
        *,
        corrected_logits: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> None:
        # Graph replay can reuse proposal outputs before a PP ring trip completes.
        self.corrected_logits = corrected_logits.detach().clone()
        self.draft_tokens = draft_tokens.detach().clone()
        self.num_rows = int(corrected_logits.shape[0])


class _PPDraftLogitsCache:
    """Last-stage cache for draft distributions relayed by compact handles."""

    def __init__(self, *, max_rows: int) -> None:
        self.max_rows = max(int(max_rows), 1)
        self._entries: OrderedDict[int, _PPDraftLogitsCacheEntry] = OrderedDict()
        self._next_id = 1
        self._num_rows = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def clear(self) -> None:
        self._entries.clear()
        self._num_rows = 0

    def put(
        self,
        *,
        corrected_logits: Optional[torch.Tensor],
        draft_tokens: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if corrected_logits is None:
            return None, None
        if corrected_logits.ndim != 3:
            raise ValueError(
                "DSpark PP cached draft logits must be rank 3, got "
                f"{tuple(corrected_logits.shape)}."
            )
        if corrected_logits.shape[:2] != draft_tokens.shape:
            raise ValueError(
                "DSpark PP cached draft logits/token shape mismatch: "
                f"logits={tuple(corrected_logits.shape)}, "
                f"tokens={tuple(draft_tokens.shape)}."
            )

        num_rows = int(corrected_logits.shape[0])
        while self._entries and self._num_rows + num_rows > self.max_rows:
            _, stale = self._entries.popitem(last=False)
            self._num_rows -= stale.num_rows
            self.evictions += 1

        cache_id = self._next_id
        self._next_id += 1
        self._entries[cache_id] = _PPDraftLogitsCacheEntry(
            corrected_logits=corrected_logits,
            draft_tokens=draft_tokens,
        )
        self._num_rows += num_rows
        return (
            torch.full((num_rows,), cache_id, dtype=torch.int64),
            torch.arange(num_rows, dtype=torch.int64),
        )

    def take(
        self,
        *,
        cache_ids,
        cache_rows,
        expected_draft_tokens: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if cache_ids is None or cache_rows is None:
            self.misses += 1
            return None
        ids = (
            cache_ids.to("cpu", dtype=torch.int64).tolist()
            if isinstance(cache_ids, torch.Tensor)
            else [int(value) for value in cache_ids]
        )
        rows = (
            cache_rows.to("cpu", dtype=torch.int64).tolist()
            if isinstance(cache_rows, torch.Tensor)
            else [int(value) for value in cache_rows]
        )
        if len(ids) != len(rows) or len(ids) != expected_draft_tokens.shape[0]:
            self.misses += 1
            return None
        if not ids or any(cache_id < 0 for cache_id in ids):
            self.misses += 1
            return None

        used_ids = set(ids)
        entries = {cache_id: self._entries.get(cache_id) for cache_id in used_ids}
        if any(entry is None for entry in entries.values()):
            self.misses += 1
            return None
        if any(
            row < 0 or row >= entries[cache_id].num_rows
            for cache_id, row in zip(ids, rows)
        ):
            self.misses += 1
            return None

        for cache_id in used_ids:
            entry = self._entries.pop(cache_id)
            self._num_rows -= entry.num_rows

        first_entry = entries[ids[0]]
        if (
            len(used_ids) == 1
            and rows == list(range(first_entry.num_rows))
            and len(rows) == first_entry.num_rows
        ):
            corrected_logits = first_entry.corrected_logits
            cached_tokens = first_entry.draft_tokens
        else:
            corrected_logits = torch.cat(
                [
                    entries[cache_id].corrected_logits[row : row + 1]
                    for cache_id, row in zip(ids, rows)
                ],
                dim=0,
            )
            cached_tokens = torch.cat(
                [
                    entries[cache_id].draft_tokens[row : row + 1]
                    for cache_id, row in zip(ids, rows)
                ],
                dim=0,
            )

        if corrected_logits.shape[:2] != expected_draft_tokens.shape:
            self.misses += 1
            return None
        if os.getenv("SGLANG_DSPARK_PP_VALIDATE_DRAFT_LOGITS_CACHE", "0") == "1":
            if not torch.equal(cached_tokens, expected_draft_tokens):
                self.misses += 1
                return None
        self.hits += 1
        return corrected_logits


class DSparkWorkerV2(BaseSpecWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.ps = ps
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.device = target_worker.device

        self._pp_is_last_rank = target_worker.pp_group.is_last_rank
        self._pp_enabled = server_args.pp_size > 1
        self._pp_retain_draft_logits = (
            self._pp_enabled
            and self._pp_is_last_rank
            and envs.SGLANG_DSPARK_PP_RETAIN_DRAFT_LOGITS.get()
        )
        self._pp_draft_logits_cache = _PPDraftLogitsCache(
            max_rows=int(server_args.max_running_requests)
        )
        if self._pp_is_last_rank and self.ps.tp_rank == 0:
            logger.info(
                "DSpark PP non-greedy verification: retain_draft_logits=%s "
                "fast_rejection=%s cache_max_rows=%d",
                self._pp_retain_draft_logits,
                envs.SGLANG_DSPARK_PP_FAST_REJECTION.get(),
                self._pp_draft_logits_cache.max_rows,
            )

        self._draft_is_moe = draft_is_deepseek_v4(server_args=server_args)
        self._draft_dp_context_enabled = (
            server_args.enable_dp_attention and not self._draft_is_moe
        )
        attn_tp_size = server_args.tp_size // max(server_args.dp_size, 1)
        if server_args.enable_dp_attention and self._draft_is_moe and attn_tp_size > 1:
            raise ValueError(
                "DSpark + dp attention with a DeepSeek-V4 (MoE) draft requires "
                "attn_tp == 1 (set --dp-size == --tp). attn_tp > 1 corrupts the "
                "MoE-under-DP all-reduce."
            )

        # The draft model is only created on the last PP rank. Non-last ranks
        # relay the target verify forward and early-return; they still need the
        # gamma / verify_num_draft_tokens constants and a planner/executor to
        # drive the shared verify-layout and target forward, so those are built
        # unconditionally below with draft_model=None on non-last ranks.
        if not self._pp_enabled or self._pp_is_last_rank:
            with self._draft_context():
                bundle = build_draft_tp_worker(
                    server_args=server_args,
                    gpu_id=gpu_id,
                    ps=replace(ps, pp_rank=0),
                    nccl_port=nccl_port,
                    target_model_config=target_worker.model_runner.model_config,
                    algo_label="DSPARK",
                    attention_backend_override=(
                        DSV4_DRAFT_ATTENTION_BACKEND if self._draft_is_moe else None
                    ),
                    pp_global_random_seed=(
                        target_worker.random_seed if self._pp_enabled else None
                    ),
                )
            self._draft_worker = bundle.draft_worker
            self.draft_model_runner = bundle.draft_model_runner
            self.draft_model = bundle.draft_model
            self._draft_sampler = None

            runtime_config = resolve_runtime_config(
                draft_hf_config=self.draft_model_runner.model_config.hf_config,
                speculative_num_draft_tokens=server_args.speculative_num_draft_tokens,
                target_vocab_size=int(
                    self.target_worker.model_runner.model_config.vocab_size
                ),
            )
            self.gamma = runtime_config.gamma
            self.verify_num_draft_tokens = runtime_config.verify_num_draft_tokens
            self.speculative_num_draft_tokens = self.verify_num_draft_tokens
            self._mask_token_id = runtime_config.mask_token_id

            if self.ps.tp_rank == 0:
                logger.info(
                    "Initialized DSpark draft runner. attention_backend=%s, model=%s, "
                    "gamma=%s, verify_num_draft_tokens=%s, mask_token_id=%s, "
                    "markov_head=%s",
                    bundle.resolved_attention_backend,
                    self.draft_model.__class__.__name__,
                    self.gamma,
                    self.verify_num_draft_tokens,
                    self._mask_token_id,
                    type(self.draft_model.markov_head).__name__,
                )

            self._block_pos_offsets = build_block_pos_offsets(
                length=self.verify_num_draft_tokens, device=self.device
            )
            self._draft_block_spec_info = make_draft_block_spec_info(
                draft_token_num=int(self.gamma), device=self.device
            )

            target_model = self.target_worker.model_runner.model
            lm_head = getattr(target_model, "lm_head", None)
            if lm_head is None or not hasattr(lm_head, "weight"):
                raise RuntimeError(
                    "DSpark requires the target model to expose `lm_head` with `weight`."
                )
            self.draft_model.attach_shared_modules(
                embed_tokens=(
                    None
                    if self._pp_enabled
                    else self._resolve_target_embed_tokens(target_model)
                ),
                lm_head=lm_head,
            )
        else:
            # Non-last PP rank: no draft model. Gamma / verify width still come
            # from server_args so alloc_verify_window dims match the last rank.
            self._draft_worker = None
            self.draft_model_runner = None
            self.draft_model = None
            self._draft_sampler = None
            self.gamma = max(server_args.speculative_num_draft_tokens - 1, 0)
            self.verify_num_draft_tokens = server_args.speculative_num_draft_tokens
            self.speculative_num_draft_tokens = self.verify_num_draft_tokens
            self._mask_token_id = 0
            self._block_pos_offsets = build_block_pos_offsets(
                length=self.verify_num_draft_tokens, device=self.device
            )
            self._draft_block_spec_info = None

        self._verify_planner = DSparkVerifyPlanner(
            draft_model=self.draft_model,
            gamma=self.gamma,
            model_runner=self.model_runner,
            device=self.device,
            tp_rank=self.ps.tp_rank,
            server_args=self.server_args,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            authoritative_scheduler=not self._pp_enabled or self._pp_is_last_rank,
        )
        if (
            server_args.enable_dp_attention
            and not self._draft_is_moe
            and self._verify_planner.is_compact_mode
            and not server_args.disable_cuda_graph
        ):
            raise ValueError(
                "DSpark dense-draft compact verify under --enable-dp-attention does not "
                "yet support cuda graph (idle DP groups cannot join the token-keyed "
                "compact graph). Re-run with --disable-cuda-graph (eager is lossless), "
                "or use SGLANG_RAGGED_VERIFY_MODE=static. The dsv4 (MoE) draft supports "
                "cuda graph under DP."
            )
        if self.draft_model is not None:
            self._kv_injector = TargetHiddenKvInjector(
                draft_model=self.draft_model,
                draft_model_runner=self.draft_model_runner,
                model_runner=self.model_runner,
                device=self.device,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                block_pos_offsets=self._block_pos_offsets,
            )
            self._proposer = DraftBlockProposer(
                draft_model=self.draft_model,
                draft_model_runner=self.draft_model_runner,
                gamma=self.gamma,
                mask_token_id=self._mask_token_id,
                draft_block_spec_info=self._draft_block_spec_info,
                dp_moe_sync=self._draft_is_moe and server_args.enable_dp_attention,
            )
        else:
            self._kv_injector = None
            self._proposer = None
        self._verify_epilogue = None
        if (
            self._verify_planner.is_compact_mode
            and not server_args.disable_cuda_graph
            and is_cuda()
            and self.draft_model is not None
        ):
            self._verify_epilogue = DsparkVerifyEpilogue(
                max_bs=max(server_args.cuda_graph_config.decode.bs),
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                device=self.device,
                # PP replays a proposal relayed through DSparkPPVerifyInputRaw.
                # Its graph-folded draft-token buffers cannot be reused safely
                # after the pipeline round trip, so accept/finalize must run
                # once outside the target graph. Keep only compact-to-strided
                # scatter in the graph and avoid recording a duplicate greedy
                # accept, TP sync, and gated commit.
                fold_accept=not self._pp_enabled,
                commit_ctx=CommitInjectCtx(
                    draft_model=self.draft_model,
                    block_pos_offsets=self._block_pos_offsets,
                    resolve_pool=lambda: self.draft_model_runner.token_to_kv_pool,
                    resolve_req_to_token=lambda: (
                        self.model_runner.req_to_token_pool.req_to_token
                    ),
                ),
            )
            self.model_runner.capture_tail_hooks.append(
                self._verify_epilogue.capture_hook
            )

        self._simulate_acc_len = float(envs.SGLANG_SIMULATE_ACC_LEN.get())
        if (
            self._simulate_acc_len > 0
            and self._simulate_acc_len != 1.0
            and not self._verify_planner.is_verify_all
        ):
            raise ValueError(
                "SGLANG_SIMULATE_ACC_LEN>1.0 with DSpark requires a verify-all "
                "schedule (SGLANG_RAGGED_VERIFY_MODE=static, or =compact with the "
                "uninitialized/flat SPS table): a constant simulated correct_len>0 "
                "can exceed a trimmed request's verify budget (cap-accept, or "
                "compact with a profiled SPS table) and break the cutoff/cap "
                "accounting. SGLANG_SIMULATE_ACC_LEN=1.0 yields correct_len=0 "
                "(commit is the bonus token only), which stays within every verify "
                "budget and is safe in any mode. Got mode="
                f"{self._verify_planner.mode_value!r}, simulate_acc_len="
                f"{self._simulate_acc_len}."
            )

        self._verify_executor = TargetVerifyExecutor(
            target_worker=self.target_worker,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            kv_injector=self._kv_injector,
            verify_epilogue=self._verify_epilogue,
            simulate_acc_len=self._simulate_acc_len,
        )

        self._forced_budget_frac: Optional[float] = None

        self._observers = DsparkStepObservers(
            planner=self._verify_planner,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_rank=self.ps.tp_rank,
            device=self.device,
            simulate_acc_len=self._simulate_acc_len,
        )

    def _resolve_target_embed_tokens(self, target_model):
        if hasattr(target_model, "get_input_embeddings"):
            return target_model.get_input_embeddings()
        return target_model.model.get_input_embeddings()

    @property
    def carries_confidence(self) -> bool:
        return self._verify_planner.carries_confidence

    @property
    def spec_v2_attn_backends(self) -> tuple:
        if self._draft_worker is None:
            return (self._target_worker.model_runner.attn_backend,)
        return (
            self._target_worker.model_runner.attn_backend,
            self.draft_model_runner.attn_backend,
        )

    def __getattr__(self, name):
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    def _draft_context(self):
        if self._draft_dp_context_enabled:
            return draft_tp_context(get_parallel().attn_tp_group)
        return nullcontext()

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        if self._draft_worker is None:
            return
        self._draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    def init_attention_backends(self):
        if self._draft_worker is None:
            return
        with self._draft_context():
            self._draft_worker.init_attention_backends()

    def init_cuda_graphs(self):
        if self._draft_worker is None:
            return
        capture_decode_cuda_graph = not self.server_args.disable_cuda_graph
        if is_cuda() and capture_decode_cuda_graph:
            available_mem = get_available_gpu_memory(self.device, self.gpu_id)
            if available_mem < 1.0:
                capture_decode_cuda_graph = False
                logger.warning(
                    "Disable DSpark draft cuda graph because only %.2f GB GPU "
                    "memory is available after target backend initialization.",
                    available_mem,
                )
        with self._draft_context():
            if capture_decode_cuda_graph:
                self._draft_sampler = self._maybe_build_draft_sampler()
                if self._draft_sampler is not None:
                    self.draft_model_runner.capture_tail_hooks.append(
                        make_draft_sampler_capture_hook(self._draft_sampler)
                    )
                self._proposer.attach_draft_sampler(self._draft_sampler)
            self._draft_worker.init_cuda_graphs(
                capture_decode_cuda_graph=capture_decode_cuda_graph
            )

    def _maybe_build_draft_sampler(self):
        return maybe_build_draft_sampler(
            draft_model=self.draft_model,
            gamma=self.gamma,
            max_bs=max(self.server_args.cuda_graph_config.decode.bs),
            device=self.device,
            tp_rank=self.ps.tp_rank,
            confidence_fn=(
                self._verify_planner.compute_confidence_tensor
                if self._verify_planner.carries_confidence
                else None
            ),
            out=(
                self._verify_epilogue.draft_tokens_buf
                if self._verify_epilogue is not None
                else None
            ),
        )

    def clear_cache_pool(self):
        self._pp_draft_logits_cache.clear()

    def _cache_pp_draft_logits(
        self,
        *,
        draft_block: DraftBlockResult,
        sampling_info,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (
            not self._pp_retain_draft_logits
            or sampling_info is None
            or sampling_info.is_all_greedy
        ):
            return None, None
        return self._pp_draft_logits_cache.put(
            corrected_logits=draft_block.corrected_logits,
            draft_tokens=draft_block.draft_tokens,
        )

    def _restore_pp_draft_logits(
        self,
        *,
        pp_raw: DSparkPPVerifyInputRaw,
        draft_tokens: torch.Tensor,
        sampling_info,
    ) -> Optional[torch.Tensor]:
        if (
            not self._pp_retain_draft_logits
            or sampling_info is None
            or sampling_info.is_all_greedy
        ):
            return None
        corrected_logits = self._pp_draft_logits_cache.take(
            cache_ids=pp_raw.draft_logits_cache_ids,
            cache_rows=pp_raw.draft_logits_cache_rows,
            expected_draft_tokens=draft_tokens,
        )
        total = self._pp_draft_logits_cache.hits + self._pp_draft_logits_cache.misses
        if self.ps.tp_rank == 0 and (
            total in (1, 10, 100) or (total > 0 and total % 1000 == 0)
        ):
            logger.info(
                "DSpark PP draft-logits cache: hits=%d misses=%d evictions=%d",
                self._pp_draft_logits_cache.hits,
                self._pp_draft_logits_cache.misses,
                self._pp_draft_logits_cache.evictions,
            )
        return corrected_logits

    def set_dspark_forced_budget_frac(self, frac: Optional[float]) -> None:
        self._forced_budget_frac = frac
        self._verify_planner.set_forced_budget_frac(frac)

    def dump_info_records(self) -> Optional[dict]:
        return self._observers.dump_info_records()

    def clear_info_records(self) -> None:
        self._observers.clear_info_records()

    def block_accept_estimate_log_suffix(self) -> Optional[str]:
        return self._observers.block_accept_estimate_log_suffix()

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        self._observers.note_request_finished(rid=rid, natural_stop=natural_stop)

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        pp_proxy_tensors=None,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise ValueError(
                "DSpark speculative decoding does not support return_logprob yet."
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            self._verify_planner.note_non_decode_step()
            self._observers.note_prefill_step()
            return self._forward_prefill(batch, on_publish, pp_proxy_tensors)

        return self._forward_decode(batch, on_publish, pp_proxy_tensors)

    def _forward_prefill(
        self, batch: ScheduleBatch, on_publish, pp_proxy_tensors=None
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_idle():
            if self.server_args.enable_dp_attention:
                batch.capture_hidden_mode = CaptureHiddenMode.FULL
                self.target_worker.forward_batch_generation(
                    batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                )
            return self._decode_idle_result(on_publish=on_publish)

        batch_output = self.target_worker.forward_batch_generation(
            batch,
            pp_proxy_tensors=pp_proxy_tensors,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        # Non-last PP rank: only relay the target prefill forward; draft KV
        # injection happens exclusively on the last rank (which holds the draft pool).
        if self._pp_enabled and not self._pp_is_last_rank:
            return batch_output

        logits_output = batch_output.logits_output
        next_token_ids = batch_output.next_token_ids
        batch_output.new_seq_lens = batch.seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        if logits_output.hidden_states is None:
            raise RuntimeError(
                "DSpark requires target aux hidden capture for prefill, but got None. "
                "Make sure the target model has DFlash layers-to-capture configured."
            )
        if batch.extend_lens is None or batch.prefix_lens is None:
            raise RuntimeError(
                "DSpark expected extend_lens / prefix_lens in extend mode, got None."
            )
        if batch.out_cache_loc is None:
            raise RuntimeError("DSpark prefill expected out_cache_loc, but got None.")

        device = next_token_ids.device
        ctx_lens = torch.tensor(batch.extend_lens, dtype=torch.int32, device=device)
        draft_seq_lens = torch.tensor(
            batch.prefix_lens, dtype=torch.int32, device=device
        )
        positions, _ = compute_position(
            self.model_runner.server_args.attention_backend,
            draft_seq_lens,
            ctx_lens,
            int(sum(batch.extend_lens)),
        )
        self._kv_injector.inject_target_hidden(
            target_hidden=logits_output.hidden_states,
            cache_loc=batch.out_cache_loc,
            positions=positions,
        )
        logits_output.hidden_states = None

        batch_output.next_draft_input = make_next_draft_input(
            bonus_tokens=next_token_ids,
            new_seq_lens=batch.seq_lens,
        )
        return batch_output

    def _idle_verify_ragged_layout(self, batch: ScheduleBatch):
        if batch.global_num_tokens is None or not self._verify_planner.is_compact_mode:
            return None
        global_bs = max(batch.global_num_tokens)
        if global_bs <= 0:
            return None
        return idle_ragged_layout(
            tier_num_reqs=global_bs,
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
        )

    def _dp_verify_tier_num_tokens(self, batch: ScheduleBatch) -> Optional[int]:
        if not (
            self._draft_is_moe
            and self.server_args.enable_dp_attention
            and batch.global_num_tokens is not None
            and self._verify_planner.is_compact_mode
        ):
            return None
        return dp_global_verify_tier_num_tokens(
            global_tier_num_tokens=batch.global_spec_verify_tier_num_tokens
        )

    def _decode_idle_result(
        self,
        *,
        on_publish,
    ) -> GenerationBatchResult:
        next_draft_input = make_next_draft_input(
            bonus_tokens=torch.empty((0,), device=self.device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
        )
        if on_publish is not None:
            on_publish(next_draft_input.new_seq_lens)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.empty((0,), dtype=torch.int64, device=self.device),
            accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            block_accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            next_draft_input=next_draft_input,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=next_draft_input.new_seq_lens,
        )

    def _draft_block_from_pp_raw(self, pp_raw, batch, sampling_info):
        device = batch.seq_lens.device
        bs = len(batch.seq_lens)
        bonus = torch.as_tensor(pp_raw.bonus_tokens, device=device, dtype=torch.int64)
        drafts = torch.as_tensor(pp_raw.draft_tokens, device=device, dtype=torch.int64)
        if bonus.shape != (bs,):
            raise RuntimeError(
                "DSpark PP bonus-token shape mismatch: "
                f"expected={(bs,)}, got={tuple(bonus.shape)}."
            )
        if drafts.shape != (bs, self.gamma):
            raise RuntimeError(
                "DSpark PP draft-token shape mismatch: "
                f"expected={(bs, self.gamma)}, got={tuple(drafts.shape)}."
            )
        draft_block_ids = bonus.unsqueeze(1)
        if sampling_info is not None:
            temperatures = (
                sampling_info.temperatures.view(-1).to(torch.float32).clamp_min(1e-5)
            )
        else:
            temperatures = torch.ones(bs, dtype=torch.float32, device=device)
        draft_block = DraftBlockResult(
            draft_tokens=drafts,
            corrected_logits=self._restore_pp_draft_logits(
                pp_raw=pp_raw,
                draft_tokens=drafts,
                sampling_info=sampling_info,
            ),
            greedy_mask=resolve_greedy_mask(
                bs=bs, sampling_info=sampling_info, device=device
            ),
            temperatures=temperatures,
        )
        confidence = (
            torch.as_tensor(pp_raw.confidence, device=device, dtype=torch.float32)
            if pp_raw.confidence is not None
            else None
        )
        if confidence is not None and confidence.shape != (bs, self.gamma):
            raise RuntimeError(
                "DSpark PP confidence shape mismatch: "
                f"expected={(bs, self.gamma)}, got={tuple(confidence.shape)}."
            )
        return draft_block_ids, draft_block, drafts, confidence

    def _forward_decode(
        self, batch: ScheduleBatch, on_publish, pp_proxy_tensors=None
    ) -> GenerationBatchResult:
        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        spec_info = batch.spec_info
        if not isinstance(spec_info, (DFlashDraftInputV2, DSparkPPVerifyInputRaw)):
            raise RuntimeError(
                "DSpark spec-v2 expected DFlashDraftInputV2 / DSparkPPVerifyInputRaw "
                "state on the running batch."
            )
        pp_raw = spec_info if isinstance(spec_info, DSparkPPVerifyInputRaw) else None
        draft_input = spec_info

        if batch.forward_mode.is_idle():
            self._observers.note_idle_decode_step()
            if self.server_args.enable_dp_attention:
                if self._draft_is_moe:
                    self._proposer.run_idle_participation(batch)
                self._verify_executor.run_idle_participation(
                    batch=batch, idle_layout=self._idle_verify_ragged_layout(batch)
                )
            return self._decode_idle_result(on_publish=on_publish)

        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        bs = len(batch.seq_lens)
        device = self.device
        prefix_lens = batch.seq_lens

        self._observers.begin_step()

        target_model = self.target_worker.model_runner.model

        verify_window = alloc_verify_window(
            batch=batch,
            bs=bs,
            device=device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
            model_runner=self.model_runner,
        )

        sampling_info = batch.sampling_info
        proposal = None
        if pp_raw is None:
            with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):
                proposal = self._proposer.propose(
                    batch=batch,
                    draft_input=draft_input,
                    verify_window=verify_window,
                    bs=bs,
                    device=device,
                    target_model=target_model,
                    sampling_info=sampling_info,
                )
            draft_block_ids = proposal.draft_block_ids
            draft_block = proposal.draft_block
            draft_tokens = draft_block.draft_tokens
            confidence = proposal.confidence
            if confidence is None:
                confidence = self._verify_planner.compute_confidence_tensor(
                    draft_hidden=proposal.draft_hidden,
                    anchor_tokens=draft_block_ids[:, 0],
                    draft_tokens=draft_tokens,
                    confidence_tap=proposal.confidence_tap,
                )
        else:
            (
                draft_block_ids,
                draft_block,
                draft_tokens,
                confidence,
            ) = self._draft_block_from_pp_raw(pp_raw, batch, sampling_info)

        global_num_reqs = (
            max(batch.global_num_tokens)
            if self._draft_is_moe
            and self.server_args.enable_dp_attention
            and batch.global_num_tokens is not None
            else None
        )
        if pp_raw is not None and not self._verify_planner.is_static_mode:
            relayed_verify_lens = pp_raw.next_verify_lens
            if relayed_verify_lens is None:
                relayed_verify_lens = [self.verify_num_draft_tokens] * bs
            verify_token_budget = self._verify_planner.verify_budget_from_lens(
                relayed_verify_lens
            )
            layout = self._verify_planner.layout_from_relayed_verify_lens(
                verify_lens=relayed_verify_lens,
                expected_bs=bs,
                device=device,
            )
        else:
            verify_token_budget = self._verify_planner.resolve_verify_token_budget(
                draft_input=draft_input,
                confidence=confidence,
                prefix_lens=prefix_lens,
                req_pool_indices=batch.req_pool_indices,
            )
            layout = self._verify_planner.schedule_layout(
                req_pool_indices=batch.req_pool_indices,
                prefix_lens=prefix_lens,
                device=device,
                confidence=confidence,
                budget=verify_token_budget,
                global_num_reqs=global_num_reqs,
                dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
            )
        run_compact = self._verify_planner.should_run_compact(layout=layout)

        verify_ids_2d = torch.cat(
            [draft_block_ids[:, :1], draft_tokens], dim=1
        ).contiguous()

        proposal_or_raw_folded = proposal.folded if proposal is not None else False
        epilogue = self._verify_executor.verify_epilogue
        fold_eligible = (
            epilogue is not None
            and epilogue.folds_accept
            and proposal_or_raw_folded
            and verify_logits_adjustments_are_noop(sampling_info)
            and self._simulate_acc_len <= 0
        )
        with self._observers.segment(InfoSegment.TARGET_VERIFY):
            if run_compact:
                target_verify, hidden_strided = self._verify_executor.run_compact(
                    batch=batch,
                    layout=layout,
                    verify_window=verify_window,
                    verify_ids_2d=verify_ids_2d,
                    draft_block_ids=draft_block_ids,
                    draft_tokens=draft_tokens,
                    bs=bs,
                    device=device,
                    sampling_info=sampling_info,
                    inject_gate=fold_eligible,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            else:
                target_verify = self._verify_executor.run_non_compact(
                    batch=batch,
                    draft_input=draft_input,
                    verify_ids_2d=verify_ids_2d,
                    verify_window=verify_window,
                    sampling_info=sampling_info,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
                hidden_strided = None
        logits_output = target_verify.logits_output
        can_run_cuda_graph = target_verify.can_run_cuda_graph

        # PP non-last rank early exit: the target verify forward is done (its
        # hidden relayed downstream via the PP ring inside TpModelWorker), so
        # skip accept/commit/propose. Relay the proxy hidden back to the
        # scheduler, which forwards it to the next PP stage.
        if self._pp_enabled and not self._pp_is_last_rank:
            pp_proxy_out = target_verify.pp_hidden_states_proxy_tensors
            assert (
                pp_proxy_out is not None
            ), "non-last PP rank must relay proxy hidden downstream"
            return GenerationBatchResult(
                pp_hidden_states_proxy_tensors=pp_proxy_out,
                next_token_ids=torch.empty((0,), dtype=torch.int64, device=device),
                can_run_cuda_graph=can_run_cuda_graph,
                speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
                pp_verify_input_raw=None,
            )

        folded_accept = fold_eligible and run_compact and can_run_cuda_graph
        accept = self._verify_executor.accept_and_finalize(
            folded_accept=folded_accept,
            bs=bs,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            draft_block=draft_block,
            sampling_info=sampling_info,
            draft_input=draft_input,
            layout=layout,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
        )
        if on_publish is not None:
            if confidence is not None:
                on_publish(accept.new_seq_lens, confidence=confidence)
            else:
                on_publish(accept.new_seq_lens)

        folded_commit = folded_accept and epilogue is not None and epilogue.folds_commit
        if not folded_commit:
            self._verify_executor.commit_hidden(
                batch=batch,
                layout=layout,
                hidden_strided=hidden_strided,
                verify_window=verify_window,
                logits_output=logits_output,
                commit_lens=accept.commit_lens,
                bs=bs,
                run_compact=run_compact,
            )
        logits_output.hidden_states = None

        self._observers.observe_verify_step(
            forward_ct=int(batch.forward_iter),
            reqs=batch.reqs,
            bs=bs,
            proposal_folded=proposal_or_raw_folded,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            layout=layout,
            confidence=confidence,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
            draft_block=draft_block,
            sampling_info=sampling_info,
            correct_len=accept.correct_len,
            cap_trim_lens=accept.cap_trim_lens,
            bonus=accept.bonus,
            commit_lens=accept.commit_lens,
            verify_token_budget=verify_token_budget,
            req_pool_indices=batch.req_pool_indices,
            verify_tier_num_tokens=int(batch.spec_verify_tier_num_tokens),
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
        )

        next_draft_input = make_next_draft_input(
            bonus_tokens=accept.bonus,
            new_seq_lens=accept.new_seq_lens,
        )
        next_draft_input.max_top_k = getattr(draft_input, "max_top_k", 1)
        next_draft_input.uniform_top_k_value = getattr(
            draft_input, "uniform_top_k_value", None
        )

        pp_raw_out = None
        if self._pp_enabled:
            next_verify_window = alloc_verify_window(
                batch=batch,
                bs=bs,
                device=device,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                block_pos_offsets=self._block_pos_offsets,
                model_runner=self.model_runner,
                prefix_lens=accept.new_seq_lens,
            )
            with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):
                proposal_next = self._proposer.propose(
                    batch=batch,
                    draft_input=next_draft_input,
                    verify_window=next_verify_window,
                    bs=bs,
                    device=device,
                    target_model=target_model,
                    sampling_info=sampling_info,
                )
            con = proposal_next.confidence
            if con is None:
                con = self._verify_planner.compute_confidence_tensor(
                    draft_hidden=proposal_next.draft_hidden,
                    anchor_tokens=proposal_next.draft_block_ids[:, 0],
                    draft_tokens=proposal_next.draft_block.draft_tokens,
                    confidence_tap=proposal_next.confidence_tap,
                )
            next_verify_lens = [self.verify_num_draft_tokens] * bs
            if not self._verify_planner.is_static_mode:
                next_verify_token_budget = (
                    self._verify_planner.compute_budget_sync(
                        confidence=con,
                        prefix_lens=accept.new_seq_lens,
                        req_pool_indices=batch.req_pool_indices,
                    )
                    if con is not None
                    else None
                )
                next_layout = self._verify_planner.schedule_layout(
                    req_pool_indices=batch.req_pool_indices,
                    prefix_lens=accept.new_seq_lens,
                    device=device,
                    confidence=con,
                    budget=next_verify_token_budget,
                )
                next_verify_lens = self._verify_planner.verify_lens_for_pp_relay(
                    layout=next_layout,
                    bs=bs,
                )
            if layout is None:
                verify_lens = torch.full(
                    (bs,),
                    self.verify_num_draft_tokens,
                    dtype=torch.int64,
                )
            elif layout.verify_lens_cpu is not None:
                verify_lens = torch.as_tensor(layout.verify_lens_cpu, dtype=torch.int64)
            else:
                verify_lens = layout.verify_lens.to("cpu", dtype=torch.int64)
            cache_ids, cache_rows = self._cache_pp_draft_logits(
                draft_block=proposal_next.draft_block,
                sampling_info=sampling_info,
            )
            pp_raw_out = DSparkPPVerifyInputRaw(
                bonus_tokens=accept.bonus,
                draft_tokens=proposal_next.draft_block.draft_tokens,
                new_seq_lens=accept.new_seq_lens,
                accept_lens=accept.commit_lens,
                max_top_k=getattr(draft_input, "max_top_k", 1),
                uniform_top_k_value=getattr(draft_input, "uniform_top_k_value", None),
                confidence=con,
                cap_trim_lens=accept.cap_trim_lens,
                verify_lens=verify_lens,
                next_verify_lens=torch.as_tensor(next_verify_lens, dtype=torch.int64),
                draft_logits_cache_ids=cache_ids,
                draft_logits_cache_rows=cache_rows,
            )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=accept.out_tokens.reshape(-1),
            accept_lens=accept.commit_lens,
            block_accept_lens=accept.commit_lens + accept.cap_trim_lens,
            cap_lens=(
                layout.verify_lens.to(torch.int32) if layout is not None else None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=accept.new_seq_lens,
            pp_verify_input_raw=pp_raw_out,
        )

    def get_confidence_budget_prepare(self):
        return self._verify_planner.confidence_budget_prepare()
