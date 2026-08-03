from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import (
    DFlashDecodePrepareMixin,
    DFlashDraftInputV2,
)
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
    compute_dflash_sampling_correct_drafts_and_bonus,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockResult,
    sync_dspark_tensor_across_tp,
)
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    VerifyWindow,
    apply_logits_adjustments_strided,
)
from sglang.srt.speculative.dspark_components.kernels.dspark_accept import (
    AcceptGreedy,
    AcceptSampling,
    FinalizeAcceptLens,
    SelectMixedAccept,
    SoftmaxTemp,
    accept_greedy_triton,
    accept_sampling_logits_fast,
    build_uniform_topk_probs,
    finalize_accept_lens_triton,
)
from sglang.srt.speculative.dspark_components.kernels.dspark_verify_window import (
    BuildCommitInjectLayout,
    BuildOutTokens,
    BuildRaggedVerifyWindow,
    RaggedVerifyWindow,
    ScatterCompactToStrided,
    scatter_compact_to_strided_into,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType

logger = logging.getLogger(__name__)
_PP_FAST_REJECTION_LOGGED = False


def _sync_accept_across_tp(
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    cap_trim_lens: torch.Tensor,
    *,
    packed_buf: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make rank zero's coherent accept outcome authoritative within TP."""
    bs = correct_len.shape[0]
    if packed_buf is None:
        packed = torch.stack(
            (
                correct_len.to(torch.int64),
                bonus.to(torch.int64),
                cap_trim_lens.to(torch.int64),
            )
        )
    else:
        if (
            packed_buf.dtype != torch.int64
            or packed_buf.ndim != 2
            or packed_buf.shape[0] != 3
            or packed_buf.shape[1] < bs
            or not packed_buf.is_contiguous()
        ):
            raise ValueError(
                "DSpark accept synchronization buffer must be contiguous "
                "int64 [3, max_bs]."
            )
        packed_buf[0, :bs].copy_(correct_len)
        packed_buf[1, :bs].copy_(bonus)
        packed_buf[2, :bs].copy_(cap_trim_lens)
        # Broadcast the fixed-size allocation. A [:, :bs] slice is
        # non-contiguous when bs < max_bs, and a temporary graph-pool tensor
        # can be recycled before an external PyNCCL node has finished with it.
        packed = packed_buf
    sync_dspark_tensor_across_tp(packed)
    return (
        packed[0, :bs].to(correct_len.dtype),
        packed[1, :bs].to(bonus.dtype),
        packed[2, :bs].to(cap_trim_lens.dtype),
    )


def verify_logits_adjustments_are_noop(sampling_info) -> bool:
    if sampling_info is None:
        return True
    if sampling_info.has_custom_logit_processor:
        return False
    if getattr(sampling_info, "acc_linear_penalties", None) is not None:
        return False
    penalizer = getattr(sampling_info, "penalizer_orchestrator", None)
    if penalizer is not None and penalizer.is_required:
        return False
    if getattr(sampling_info, "vocab_mask", None) is not None:
        return False
    if getattr(sampling_info, "logit_bias", None) is not None:
        return False
    return True


class TargetVerifyResult(msgspec.Struct, frozen=True):
    logits_output: object
    can_run_cuda_graph: bool
    # PP non-last rank: the proxy hidden relayed downstream via the PP ring.
    pp_hidden_states_proxy_tensors: Optional[object] = None


@dataclass
class DSparkPPVerifyInputRaw(DFlashDecodePrepareMixin, SpecInput):
    """Serializable DSpark state relayed around the pipeline output ring."""

    bonus_tokens: List[int] | torch.Tensor
    draft_tokens: List[List[int]] | torch.Tensor
    new_seq_lens: List[int] | torch.Tensor
    accept_lens: List[int] | torch.Tensor
    max_top_k: int = 1
    uniform_top_k_value: Optional[int] = None
    reserved_seq_lens_cpu: Optional[List[int] | torch.Tensor] = None
    reserved_seq_lens_sum: Optional[int] = None
    confidence: Optional[List[List[float]] | torch.Tensor] = None
    cap_trim_lens: Optional[List[int] | torch.Tensor] = None
    verify_lens: Optional[List[int] | torch.Tensor] = None
    next_verify_lens: Optional[List[int] | torch.Tensor] = None
    accept_index: Optional[List | torch.Tensor] = None
    draft_logits_cache_ids: Optional[List[int] | torch.Tensor] = None
    draft_logits_cache_rows: Optional[List[int] | torch.Tensor] = None

    def __post_init__(self) -> None:
        super().__init__(SpecInputType.DFLASH_PP_VERIFY_INPUT_RAW)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return (1, 1)

    @staticmethod
    def _to_list(value):
        if isinstance(value, torch.Tensor):
            return DSparkPPVerifyInputRaw._to_list(value.tolist())
        if isinstance(value, dict):
            return {
                key: DSparkPPVerifyInputRaw._to_list(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [DSparkPPVerifyInputRaw._to_list(item) for item in value]
        return value

    @staticmethod
    def _as_tensor(value, *, dtype: torch.dtype, device=None):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype)
        return torch.tensor(value, dtype=dtype, device=device)

    def to_tensor_dict(self, device=None) -> dict:
        payload = {
            "dspark_pp_spec_version": 2,
            "dspark_pp_bonus_tokens": self._as_tensor(
                self.bonus_tokens, dtype=torch.int64, device=device
            ),
            "dspark_pp_draft_tokens": self._as_tensor(
                self.draft_tokens, dtype=torch.int64, device=device
            ),
            "dspark_pp_new_seq_lens": self._as_tensor(
                self.new_seq_lens, dtype=torch.int64, device=device
            ),
            "dspark_pp_accept_lens": self._as_tensor(
                self.accept_lens, dtype=torch.int64, device=device
            ),
            "dspark_pp_max_top_k": self.max_top_k,
            "dspark_pp_uniform_top_k_value": self.uniform_top_k_value,
            "dspark_pp_reserved_seq_lens_sum": self.reserved_seq_lens_sum,
            "dspark_pp_accept_index": self._to_list(self.accept_index),
        }
        optional_tensors = (
            ("reserved_seq_lens_cpu", self.reserved_seq_lens_cpu, torch.int64),
            ("confidence", self.confidence, torch.float32),
            ("cap_trim_lens", self.cap_trim_lens, torch.int64),
            ("verify_lens", self.verify_lens, torch.int64),
            ("next_verify_lens", self.next_verify_lens, torch.int64),
            ("draft_logits_cache_ids", self.draft_logits_cache_ids, torch.int64),
            ("draft_logits_cache_rows", self.draft_logits_cache_rows, torch.int64),
        )
        for name, value, dtype in optional_tensors:
            if value is not None:
                payload[f"dspark_pp_{name}"] = self._as_tensor(
                    value, dtype=dtype, device=device
                )
        return payload

    def to_serializable_dict(self) -> dict:
        """Return one CPU metadata object for the PP output ring.

        DSpark's per-request relay is tiny. Serializing it as one object avoids
        overlapping a variable number of asynchronous GPU tensor sends with
        the next PP proxy message when multiple microbatches are active.
        """
        return {
            "pp_spec_output": {
                field.name: self._to_list(getattr(self, field.name))
                for field in fields(self)
            }
        }

    @classmethod
    def from_pp_outputs(cls, pp_outputs):
        tensors = pp_outputs.tensors if hasattr(pp_outputs, "tensors") else pp_outputs
        if "dspark_pp_spec_version" in tensors:
            return cls(
                bonus_tokens=tensors["dspark_pp_bonus_tokens"],
                draft_tokens=tensors["dspark_pp_draft_tokens"],
                new_seq_lens=tensors["dspark_pp_new_seq_lens"],
                accept_lens=tensors["dspark_pp_accept_lens"],
                max_top_k=tensors.get("dspark_pp_max_top_k", 1),
                uniform_top_k_value=tensors.get("dspark_pp_uniform_top_k_value"),
                reserved_seq_lens_cpu=tensors.get("dspark_pp_reserved_seq_lens_cpu"),
                reserved_seq_lens_sum=tensors.get("dspark_pp_reserved_seq_lens_sum"),
                confidence=tensors.get("dspark_pp_confidence"),
                cap_trim_lens=tensors.get("dspark_pp_cap_trim_lens"),
                verify_lens=tensors.get("dspark_pp_verify_lens"),
                next_verify_lens=tensors.get("dspark_pp_next_verify_lens"),
                accept_index=tensors.get("dspark_pp_accept_index"),
                draft_logits_cache_ids=tensors.get("dspark_pp_draft_logits_cache_ids"),
                draft_logits_cache_rows=tensors.get(
                    "dspark_pp_draft_logits_cache_rows"
                ),
            )
        return cls(**tensors["pp_spec_output"])

    @staticmethod
    def is_in_tensor_dict(tensors: dict) -> bool:
        return "dspark_pp_spec_version" in tensors or "pp_spec_output" in tensors

    @classmethod
    def build_dummy_for_decode(cls, batch, num_draft: int) -> DSparkPPVerifyInputRaw:
        bs = len(batch.reqs)
        gamma = max(num_draft - 1, 0)
        bonus = batch.input_ids.to(torch.int64)
        return cls(
            bonus_tokens=bonus,
            draft_tokens=bonus[:, None].expand(bs, gamma).clone(),
            new_seq_lens=batch.seq_lens.to(torch.int64),
            confidence=torch.zeros(
                (bs, gamma), dtype=torch.float32, device=bonus.device
            ),
            accept_lens=torch.ones(bs, dtype=torch.int64, device=bonus.device),
            cap_trim_lens=torch.zeros(bs, dtype=torch.int64, device=bonus.device),
            verify_lens=torch.full(
                (bs,), num_draft, dtype=torch.int64, device=bonus.device
            ),
            next_verify_lens=torch.full(
                (bs,), num_draft, dtype=torch.int64, device=bonus.device
            ),
        )

    def filter_batch(self, new_indices, new_indices_cpu: Optional[List[int]] = None):
        raw_indices = new_indices_cpu if new_indices_cpu is not None else new_indices
        indices = None

        def list_indices():
            nonlocal indices
            if indices is None:
                if isinstance(raw_indices, torch.Tensor):
                    indices = raw_indices.tolist()
                else:
                    indices = [int(index) for index in raw_indices]
            return indices

        def pick(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                source_indices = (
                    new_indices
                    if isinstance(new_indices, torch.Tensor)
                    else raw_indices
                )
                index = torch.as_tensor(
                    source_indices, dtype=torch.long, device=value.device
                )
                return value.index_select(0, index)
            return [value[index] for index in list_indices()]

        self.bonus_tokens = pick(self.bonus_tokens)
        self.draft_tokens = pick(self.draft_tokens)
        self.new_seq_lens = pick(self.new_seq_lens)
        self.accept_lens = pick(self.accept_lens)
        self.reserved_seq_lens_cpu = pick(self.reserved_seq_lens_cpu)
        self.confidence = pick(self.confidence)
        self.cap_trim_lens = pick(self.cap_trim_lens)
        self.verify_lens = pick(self.verify_lens)
        self.next_verify_lens = pick(self.next_verify_lens)
        self.accept_index = pick(self.accept_index)
        self.draft_logits_cache_ids = pick(self.draft_logits_cache_ids)
        self.draft_logits_cache_rows = pick(self.draft_logits_cache_rows)
        if self.reserved_seq_lens_cpu is not None:
            self.reserved_seq_lens_sum = int(
                torch.as_tensor(self.reserved_seq_lens_cpu).sum().item()
            )
        else:
            self.reserved_seq_lens_sum = None

    def merge_batch(self, other: DSparkPPVerifyInputRaw):
        def is_empty(value) -> bool:
            return value is None or len(value) == 0

        def copy_value(value):
            return value if isinstance(value, torch.Tensor) else self._to_list(value)

        if is_empty(other.bonus_tokens):
            return
        if is_empty(self.bonus_tokens):
            self.bonus_tokens = copy_value(other.bonus_tokens)
            self.draft_tokens = copy_value(other.draft_tokens)
            self.new_seq_lens = copy_value(other.new_seq_lens)
            self.accept_lens = copy_value(other.accept_lens)
            self.max_top_k = other.max_top_k
            self.uniform_top_k_value = other.uniform_top_k_value
            self.reserved_seq_lens_cpu = copy_value(other.reserved_seq_lens_cpu)
            self.reserved_seq_lens_sum = other.reserved_seq_lens_sum
            self.confidence = copy_value(other.confidence)
            self.cap_trim_lens = copy_value(other.cap_trim_lens)
            self.verify_lens = copy_value(other.verify_lens)
            self.next_verify_lens = copy_value(other.next_verify_lens)
            self.accept_index = copy_value(other.accept_index)
            self.draft_logits_cache_ids = copy_value(other.draft_logits_cache_ids)
            self.draft_logits_cache_rows = copy_value(other.draft_logits_cache_rows)
            return

        def merge_required(lhs, rhs):
            if isinstance(lhs, torch.Tensor) or isinstance(rhs, torch.Tensor):
                template = lhs if isinstance(lhs, torch.Tensor) else rhs
                lhs_tensor = torch.as_tensor(
                    lhs, dtype=template.dtype, device=template.device
                )
                rhs_tensor = torch.as_tensor(
                    rhs, dtype=template.dtype, device=template.device
                )
                return torch.cat((lhs_tensor, rhs_tensor), dim=0)
            return self._to_list(lhs) + self._to_list(rhs)

        def merge_optional(lhs, rhs):
            if lhs is None or rhs is None:
                return None
            return merge_required(lhs, rhs)

        self.bonus_tokens = merge_required(self.bonus_tokens, other.bonus_tokens)
        self.draft_tokens = merge_required(self.draft_tokens, other.draft_tokens)
        self.new_seq_lens = merge_required(self.new_seq_lens, other.new_seq_lens)
        self.accept_lens = merge_required(self.accept_lens, other.accept_lens)
        self.max_top_k = max(self.max_top_k, other.max_top_k)
        if self.uniform_top_k_value != other.uniform_top_k_value:
            self.uniform_top_k_value = None
        self.reserved_seq_lens_cpu = merge_optional(
            self.reserved_seq_lens_cpu, other.reserved_seq_lens_cpu
        )
        self.confidence = merge_optional(self.confidence, other.confidence)
        self.cap_trim_lens = merge_optional(self.cap_trim_lens, other.cap_trim_lens)
        self.verify_lens = merge_optional(self.verify_lens, other.verify_lens)
        self.next_verify_lens = merge_optional(
            self.next_verify_lens, other.next_verify_lens
        )
        self.accept_index = merge_optional(self.accept_index, other.accept_index)
        self.draft_logits_cache_ids = merge_optional(
            self.draft_logits_cache_ids, other.draft_logits_cache_ids
        )
        self.draft_logits_cache_rows = merge_optional(
            self.draft_logits_cache_rows, other.draft_logits_cache_rows
        )
        if self.reserved_seq_lens_cpu is not None:
            self.reserved_seq_lens_sum = int(
                torch.as_tensor(self.reserved_seq_lens_cpu).sum().item()
            )
        else:
            self.reserved_seq_lens_sum = None


class TargetVerifyExecutor:
    def __init__(
        self,
        *,
        target_worker,
        gamma: int,
        verify_num_draft_tokens: int,
        model_runner,
        kv_injector: TargetHiddenKvInjector,
        verify_epilogue=None,
        simulate_acc_len: float = 0.0,
    ) -> None:
        self.target_worker = target_worker
        self.gamma = int(gamma)
        self.verify_num_draft_tokens = verify_num_draft_tokens
        self.model_runner = model_runner
        self.kv_injector = kv_injector
        self.verify_epilogue = verify_epilogue
        self._verify_backend_self_adds_seq_lens_cache: Optional[bool] = None
        self._simulate_acc_len = float(simulate_acc_len)
        self._simulated_correct_drafts_buf: Optional[torch.Tensor] = None

    def accept_and_finalize(
        self,
        *,
        folded_accept: bool,
        bs: int,
        verify_ids_2d: torch.Tensor,
        target_logits: Optional[torch.Tensor],
        draft_block: DraftBlockResult,
        sampling_info,
        draft_input: DFlashDraftInputV2,
        layout: Optional[RaggedVerifyLayout],
        prefix_lens: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> AcceptOuts:
        """Produce the per-request accept outcome after target verify.

        Folded path: the accept/finalize/out-token kernels already ran inside
        the target-verify cuda graph (DsparkVerifyEpilogue); read its buffers.
        Eager path: run them here, including the SGLANG_SIMULATE_ACC_LEN
        override.
        """
        if folded_accept:
            return self.verify_epilogue.read_accept(bs)

        correct_len, bonus, cap_trim_lens = accept_draft_tokens(
            candidates=verify_ids_2d,
            target_logits=target_logits,
            draft_block=draft_block,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            cutoff_layout=layout,
        )
        if self._simulate_acc_len > 0:
            correct_len = self._simulated_correct_len(
                bs=bs, dtype=correct_len.dtype, device=correct_len.device
            )
        correct_len, bonus, cap_trim_lens = _sync_accept_across_tp(
            correct_len,
            bonus,
            cap_trim_lens,
        )

        finalized = FinalizeAcceptLens.execute(
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=prefix_lens,
        )
        out_tokens = BuildOutTokens.execute(
            draft_tokens=draft_tokens,
            correct_len=correct_len,
            bonus=bonus,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            gamma=self.gamma,
        )
        return AcceptOuts(
            correct_len=correct_len,
            bonus=bonus,
            cap_trim_lens=finalized.cap_trim_lens,
            commit_lens=finalized.commit_lens,
            new_seq_lens=finalized.new_seq_lens,
            out_tokens=out_tokens,
        )

    def _simulated_correct_len(
        self, *, bs: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        buf = self._simulated_correct_drafts_buf
        if buf is None or buf.numel() < bs or buf.dtype != dtype:
            correct_target = int(
                round(min(max(self._simulate_acc_len - 1.0, 0.0), float(self.gamma)))
            )
            buf = torch.full(
                (max(bs, 512),), correct_target, dtype=dtype, device=device
            )
            self._simulated_correct_drafts_buf = buf
        return buf[:bs]

    def run_idle_participation(
        self,
        *,
        batch: ScheduleBatch,
        idle_layout: Optional[RaggedVerifyLayout],
    ) -> None:
        """Run a dummy target-verify forward so an idle DP rank joins the
        token-keyed collective ops of the busy ranks' verify step."""
        device = self.model_runner.device
        if self.verify_epilogue is not None:
            self.verify_epilogue.begin_step(None, armed=False)
        num_dummy_tokens = (
            idle_layout.graph_num_tokens if idle_layout is not None else 0
        )
        verify_input = DFlashVerifyInput(
            draft_token=torch.zeros(
                (num_dummy_tokens,), dtype=torch.int64, device=device
            ),
            positions=torch.zeros(
                (num_dummy_tokens,), dtype=torch.int64, device=device
            ),
            draft_token_num=self.verify_num_draft_tokens,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            ragged_verify_layout=idle_layout,
        )
        batch.out_cache_loc = torch.zeros(
            (num_dummy_tokens,), dtype=torch.int64, device=device
        )
        if idle_layout is not None:
            num_dummy_slots = int(idle_layout.verify_lens.numel())
            batch.seq_lens = torch.ones(
                (num_dummy_slots,), dtype=torch.int64, device=device
            )
            batch.req_pool_indices = torch.zeros(
                (num_dummy_slots,), dtype=torch.int64, device=device
            )
            batch.seq_lens_cpu = torch.ones((num_dummy_slots,), dtype=torch.int64)
            batch.seq_lens_sum = num_dummy_slots
            batch.forward_mode = ForwardMode.TARGET_VERIFY
        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker
        )
        self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )

    def run_non_compact(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_ids_2d: torch.Tensor,
        verify_window: VerifyWindow,
        sampling_info,
        pp_proxy_tensors=None,
    ) -> TargetVerifyResult:
        verify_w = self.verify_num_draft_tokens
        positions_2d = verify_window.positions_2d
        verify_cache_loc = verify_window.verify_cache_loc

        verify_input = DFlashVerifyInput(
            draft_token=verify_ids_2d.reshape(-1),
            positions=positions_2d.reshape(-1),
            draft_token_num=verify_w,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        batch.out_cache_loc = verify_cache_loc
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if not self._verify_backend_self_adds_seq_lens():
            if seq_lens_cpu_backup is not None:
                batch.seq_lens_cpu = seq_lens_cpu_backup + verify_w
                batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            elif draft_input.reserved_seq_lens_cpu is not None:
                batch.seq_lens_cpu = draft_input.reserved_seq_lens_cpu
                batch.seq_lens_sum = int(draft_input.reserved_seq_lens_sum)

        result = self._forward_prepared_verify(
            batch=batch,
            verify_input=verify_input,
            seq_lens_cpu_backup=seq_lens_cpu_backup,
            seq_lens_sum_backup=seq_lens_sum_backup,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        # Logits adjustments are sampling-time corrections (penalizer / vocab
        # mask / logit bias). Only the PP last rank produces logits; non-last
        # ranks relay hidden states downstream and never sample, so their
        # result.logits_output is None and must skip this.
        if result.logits_output is not None:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=result.logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=verify_w,
            )

        return result

    def _forward_prepared_verify(
        self,
        *,
        batch: ScheduleBatch,
        verify_input: DFlashVerifyInput,
        seq_lens_cpu_backup,
        seq_lens_sum_backup,
        pp_proxy_tensors=None,
    ) -> TargetVerifyResult:
        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker
        )
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.seq_lens_sum = seq_lens_sum_backup

        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return TargetVerifyResult(
            logits_output=target_out.logits_output,
            can_run_cuda_graph=target_out.can_run_cuda_graph,
            pp_hidden_states_proxy_tensors=getattr(
                target_out, "pp_hidden_states_proxy_tensors", None
            ),
        )

    def commit_hidden(
        self,
        *,
        batch: ScheduleBatch,
        layout: Optional[RaggedVerifyLayout],
        hidden_strided: Optional[torch.Tensor],
        verify_window: VerifyWindow,
        logits_output,
        commit_lens: torch.Tensor,
        bs: int,
        run_compact: bool,
    ) -> None:
        if run_compact:
            self.kv_injector.inject_ragged(
                batch=batch,
                layout=layout,
                hidden_strided=hidden_strided,
                commit_lens=commit_lens,
                bs=bs,
            )
            return
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DSpark verify requires target hidden states, got None.")
        hidden = hidden.view(bs, self.verify_num_draft_tokens, -1)
        self.kv_injector.inject_target_hidden(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=verify_window.verify_cache_loc,
            cache_loc_2d=verify_window.verify_cache_loc_2d,
            positions=verify_window.positions_2d.reshape(-1),
            commit_lens=commit_lens,
        )

    def _run_ragged(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        ragged_window: RaggedVerifyWindow,
        sampling_info,
        pp_proxy_tensors=None,
    ) -> TargetVerifyResult:
        verify_input = DFlashVerifyInput(
            draft_token=ragged_window.verify_ids,
            positions=ragged_window.positions,
            draft_token_num=self.verify_num_draft_tokens,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            ragged_verify_layout=layout,
        )
        batch.out_cache_loc = ragged_window.verify_cache_loc
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if not self._verify_backend_self_adds_seq_lens():
            if seq_lens_cpu_backup is not None:
                verify_lens_cpu = (
                    layout.verify_lens_cpu
                    if layout.verify_lens_cpu is not None
                    else layout.verify_lens.cpu().tolist()
                )
                batch.seq_lens_cpu = seq_lens_cpu_backup + torch.tensor(
                    verify_lens_cpu, dtype=seq_lens_cpu_backup.dtype
                )
                batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())

        return self._forward_prepared_verify(
            batch=batch,
            verify_input=verify_input,
            seq_lens_cpu_backup=seq_lens_cpu_backup,
            seq_lens_sum_backup=seq_lens_sum_backup,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def run_compact(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        verify_window: VerifyWindow,
        verify_ids_2d: torch.Tensor,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        bs: int,
        device: str,
        sampling_info,
        inject_gate: bool = False,
        pp_proxy_tensors=None,
    ) -> tuple[TargetVerifyResult, Optional[torch.Tensor]]:
        stride = self.verify_num_draft_tokens
        full_tokens = bs * stride
        verify_lens_cpu = layout.verify_lens_cpu
        full_width = (
            verify_lens_cpu is not None
            and len(verify_lens_cpu) == bs
            and all(int(length) == stride for length in verify_lens_cpu)
            and layout.total_verify_tokens == full_tokens
            and layout.graph_num_tokens == full_tokens
        )
        if full_width:
            assert tuple(verify_ids_2d.shape) == (bs, stride)
            ragged_window = RaggedVerifyWindow(
                positions=verify_window.positions_2d.reshape(-1),
                verify_cache_loc=verify_window.verify_cache_loc,
                verify_ids=verify_ids_2d.reshape(-1),
            )
        else:
            ragged_window = BuildRaggedVerifyWindow.execute(
                batch=batch,
                layout=layout,
                draft_block_ids=draft_block_ids,
                draft_tokens=draft_tokens,
                bs=bs,
                device=device,
                verify_num_draft_tokens=stride,
                model_runner=self.model_runner,
            )
        if self.verify_epilogue is not None:
            self.verify_epilogue.begin_step(layout.verify_lens, armed=inject_gate)
        target_verify = self._run_ragged(
            batch=batch,
            layout=layout,
            ragged_window=ragged_window,
            sampling_info=sampling_info,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        logits_output = target_verify.logits_output
        if logits_output is None:
            return target_verify, None

        if self.verify_epilogue is not None and target_verify.can_run_cuda_graph:
            strided_logits = self.verify_epilogue.strided_logits
            hidden_strided = self.verify_epilogue.strided_hidden
            assert strided_logits is not None and hidden_strided is not None, (
                "verify epilogue buffers unwritten after a graph replay -- the "
                "replayed graph was captured without the epilogue"
            )
            strided_logits = strided_logits[: bs * stride]
            hidden_strided = hidden_strided[: bs * stride]
        else:
            compact_logits = logits_output.next_token_logits
            strided_logits = ScatterCompactToStrided.execute(
                compact=compact_logits,
                layout=layout,
                fill_value=0.0,
                verify_num_draft_tokens=stride,
            )
            compact_hidden = logits_output.hidden_states
            if compact_hidden is None:
                raise RuntimeError(
                    "DSpark verify requires target hidden states, got None."
                )
            hidden_strided = ScatterCompactToStrided.execute(
                compact=compact_hidden,
                layout=layout,
                fill_value=0.0,
                verify_num_draft_tokens=stride,
            )
        apply_logits_adjustments_strided(
            next_token_logits=strided_logits,
            sampling_info=sampling_info,
            verify_num_draft_tokens=stride,
        )
        logits_output.next_token_logits = strided_logits
        logits_output.hidden_states = hidden_strided
        return target_verify, hidden_strided

    def _verify_backend_self_adds_seq_lens(self) -> bool:
        if self._verify_backend_self_adds_seq_lens_cache is None:
            backend = self.target_worker.model_runner.attn_backend
            self._verify_backend_self_adds_seq_lens_cache = hasattr(
                backend, "make_forward_metadata_from_raw_verify"
            )
        return self._verify_backend_self_adds_seq_lens_cache


class CommitInjectCtx(msgspec.Struct):

    draft_model: object
    block_pos_offsets: torch.Tensor
    resolve_pool: object
    resolve_req_to_token: object


class AcceptOuts(msgspec.Struct):
    correct_len: torch.Tensor
    bonus: torch.Tensor
    cap_trim_lens: torch.Tensor
    commit_lens: torch.Tensor
    new_seq_lens: torch.Tensor
    out_tokens: torch.Tensor


class DsparkVerifyEpilogue:

    def __init__(
        self,
        *,
        max_bs: int,
        verify_num_draft_tokens: int,
        device,
        commit_ctx: Optional[CommitInjectCtx] = None,
        fold_accept: bool = True,
    ) -> None:
        self.max_bs = int(max_bs)
        self.stride = int(verify_num_draft_tokens)
        self.gamma = self.stride - 1
        self.commit_ctx = commit_ctx
        # This controls capture-time graph topology. Keep it private and
        # expose only a read-only capability below so replay-time dispatch
        # cannot diverge from the graph that was actually recorded.
        self._fold_accept = bool(fold_accept)
        self.inject_gate_buf = torch.zeros((1,), dtype=torch.int32, device=device)
        self.verify_lens_buf = torch.zeros(
            (self.max_bs,), dtype=torch.int64, device=device
        )
        self.draft_tokens_buf = torch.zeros(
            (self.max_bs * self.gamma,), dtype=torch.int64, device=device
        )
        self.correct_len_buf = torch.zeros(
            (self.max_bs,), dtype=torch.int64, device=device
        )
        self.accept_sync_buf = torch.zeros(
            (3, self.max_bs), dtype=torch.int64, device=device
        )
        self.bonus_buf = torch.zeros((self.max_bs,), dtype=torch.int64, device=device)
        self.cap_trim_lens_buf = torch.zeros(
            (self.max_bs,), dtype=torch.int32, device=device
        )
        self.commit_lens_buf = torch.zeros(
            (self.max_bs,), dtype=torch.int32, device=device
        )
        self.new_seq_lens_buf = torch.zeros(
            (self.max_bs,), dtype=torch.int64, device=device
        )
        self.out_tokens_buf = torch.zeros(
            (self.max_bs, self.stride), dtype=torch.int64, device=device
        )
        self.strided_logits: Optional[torch.Tensor] = None
        self.strided_hidden: Optional[torch.Tensor] = None

    def capture_hook(self, runner, out, forward_batch, num_tokens) -> None:
        if runner.model_runner.is_draft_worker or not runner.ragged_verify_mode:
            return
        if (
            not isinstance(out, LogitsProcessorOutput)
            or out.next_token_logits is None
            or out.hidden_states is None
        ):
            return
        self(
            compact_logits=out.next_token_logits,
            compact_hidden=out.hidden_states,
            input_ids=forward_batch.input_ids,
            seq_lens=forward_batch.seq_lens,
            req_pool_indices=forward_batch.req_pool_indices,
            bs=forward_batch.batch_size,
        )

    def begin_step(self, verify_lens, armed: bool) -> None:
        if verify_lens is None:
            self.verify_lens_buf.zero_()
        else:
            bs = verify_lens.shape[0]
            self.verify_lens_buf[:bs].copy_(verify_lens)
            if bs < self.max_bs:
                self.verify_lens_buf[bs:].zero_()
        self.inject_gate_buf.fill_(1 if armed else 0)

    def read_accept(self, bs: int) -> AcceptOuts:
        return AcceptOuts(
            correct_len=self.correct_len_buf[:bs],
            bonus=self.bonus_buf[:bs],
            cap_trim_lens=self.cap_trim_lens_buf[:bs],
            commit_lens=self.commit_lens_buf[:bs],
            new_seq_lens=self.new_seq_lens_buf[:bs],
            out_tokens=self.out_tokens_buf[:bs],
        )

    @property
    def folds_accept(self) -> bool:
        return self._fold_accept

    @property
    def folds_commit(self) -> bool:
        if not self.folds_accept or self.commit_ctx is None:
            return False
        pool = self.commit_ctx.resolve_pool()
        return hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope")

    def _ensure_out(
        self, buf: Optional[torch.Tensor], compact: torch.Tensor
    ) -> torch.Tensor:
        if (
            buf is not None
            and buf.dtype == compact.dtype
            and buf.shape[1] == compact.shape[1]
        ):
            return buf
        assert not torch.cuda.is_current_stream_capturing(), (
            "DsparkVerifyEpilogue output buffers must be allocated during "
            "warmup, not inside graph capture (pool memory is unreadable "
            "post-replay)."
        )
        return torch.empty(
            (self.max_bs * self.stride, compact.shape[1]),
            dtype=compact.dtype,
            device=compact.device,
        )

    def __call__(
        self,
        *,
        compact_logits: torch.Tensor,
        compact_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        bs: int,
    ) -> None:
        self.strided_logits = self._ensure_out(self.strided_logits, compact_logits)
        self.strided_hidden = self._ensure_out(self.strided_hidden, compact_hidden)
        verify_lens = self.verify_lens_buf[:bs]
        self._scatter(compact_logits, compact_hidden, verify_lens, bs)
        if not self.folds_accept:
            return
        commit_lens = self._accept(input_ids, seq_lens, verify_lens, bs)
        if self.folds_commit:
            self._commit_inject(
                commit_lens, verify_lens, seq_lens, req_pool_indices, bs
            )

    def _scatter(self, compact_logits, compact_hidden, verify_lens, bs: int) -> None:
        verify_lens = verify_lens.to(dtype=torch.int64).contiguous()
        start = (torch.cumsum(verify_lens, dim=0) - verify_lens).contiguous()
        # A ragged graph tier can contain zero-length padding request slots.
        # Scatter-only PP returns just the real-bs prefix and has no captured
        # accept/commit consumer, so preserve those blocks instead of issuing
        # vocab-wide fill stores. Folded accept keeps the fully-defined output.
        skip_zero_lens = not self.folds_accept
        scatter_compact_to_strided_into(
            compact=compact_logits,
            verify_lens=verify_lens,
            out=self.strided_logits[: bs * self.stride],
            stride=self.stride,
            fill_value=0.0,
            start=start,
            skip_zero_lens=skip_zero_lens,
        )
        scatter_compact_to_strided_into(
            compact=compact_hidden,
            verify_lens=verify_lens,
            out=self.strided_hidden[: bs * self.stride],
            stride=self.stride,
            fill_value=0.0,
            start=start,
            skip_zero_lens=skip_zero_lens,
        )

    def _accept(self, input_ids, seq_lens, verify_lens, bs: int) -> torch.Tensor:
        candidates = torch.zeros(
            (bs * self.stride, 1), dtype=input_ids.dtype, device=input_ids.device
        )
        scatter_compact_to_strided_into(
            compact=input_ids.view(-1, 1),
            verify_lens=verify_lens,
            out=candidates,
            stride=self.stride,
            fill_value=0,
        )
        correct_len, bonus, cap_trim_lens = accept_greedy_triton(
            candidates=candidates.view(bs, self.stride),
            target_logits=self.strided_logits[: bs * self.stride],
            verify_num_draft_tokens=self.stride,
            cutoff_verify_lens=verify_lens,
        )
        correct_len, bonus, cap_trim_lens = _sync_accept_across_tp(
            correct_len,
            bonus,
            cap_trim_lens,
            packed_buf=self.accept_sync_buf,
        )
        finalized = finalize_accept_lens_triton(
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=seq_lens[:bs],
        )
        out_tokens = BuildOutTokens.execute(
            draft_tokens=self.draft_tokens_buf[: bs * self.gamma].view(bs, self.gamma),
            correct_len=correct_len,
            bonus=bonus,
            verify_num_draft_tokens=self.stride,
            gamma=self.gamma,
        )
        self.correct_len_buf[:bs].copy_(correct_len)
        self.bonus_buf[:bs].copy_(bonus)
        self.cap_trim_lens_buf[:bs].copy_(cap_trim_lens.to(torch.int32))
        self.commit_lens_buf[:bs].copy_(finalized.commit_lens)
        self.new_seq_lens_buf[:bs].copy_(finalized.new_seq_lens)
        self.out_tokens_buf[:bs].copy_(out_tokens.view(bs, self.stride))
        return finalized.commit_lens

    def _commit_inject(
        self, commit_lens, verify_lens, seq_lens, req_pool_indices, bs: int
    ) -> None:
        ctx = self.commit_ctx
        pool = ctx.resolve_pool()
        gated_commit_lens = (
            torch.minimum(commit_lens, verify_lens.to(torch.int32))
            * self.inject_gate_buf
        )
        inject_layout = BuildCommitInjectLayout.execute(
            req_pool_indices=req_pool_indices,
            req_to_token=ctx.resolve_req_to_token(),
            prefix_lens=seq_lens[:bs],
            block_pos_offsets=ctx.block_pos_offsets[: self.stride],
            full_to_swa_mapping=pool.full_to_swa_index_mapping,
            commit_lens=gated_commit_lens,
            stride=self.stride,
        )
        with torch.inference_mode():
            ctx.draft_model.write_target_hidden_kv(
                main_hidden=self.strided_hidden[: bs * self.stride],
                swa_loc=inject_layout.swa_loc,
                positions=inject_layout.positions,
                pool=pool,
            )


def accept_draft_tokens(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_block: DraftBlockResult,
    sampling_info,
    draft_input: DFlashDraftInputV2 | DSparkPPVerifyInputRaw,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_layout: Optional[RaggedVerifyLayout] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _PP_FAST_REJECTION_LOGGED

    greedy_mask = draft_block.greedy_mask
    cutoff_verify_lens = None if cutoff_layout is None else cutoff_layout.verify_lens
    all_greedy = sampling_info is None or sampling_info.is_all_greedy
    if all_greedy:
        return AcceptGreedy.execute(
            candidates=candidates,
            target_logits=target_logits,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )
    if draft_block.corrected_logits is None:
        if not isinstance(draft_input, DSparkPPVerifyInputRaw):
            raise RuntimeError(
                "DSpark non-greedy verification requires draft logits outside PP."
            )
        # A missing or evicted cache entry keeps progressing through target-only
        # verification, at the cost of a lower acceptance rate.
        return _accept_target_only_draft_tokens(
            candidates=candidates,
            target_logits=target_logits,
            sampling_info=sampling_info,
            draft_input=draft_input,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )
    draft_top_k = None
    uniform_top_k_value = getattr(draft_input, "uniform_top_k_value", None)
    max_top_k = getattr(draft_input, "max_top_k", None)
    if (
        envs.SGLANG_DSPARK_DRAFT_TOPK_SAMPLING.get()
        and sampling_info.need_top_k_sampling
        and not sampling_info.need_top_p_sampling
        and not getattr(sampling_info, "need_min_p_sampling", False)
        and uniform_top_k_value is not None
        and int(uniform_top_k_value) == int(max_top_k or -1)
        and 1 < int(uniform_top_k_value) < int(draft_block.corrected_logits.shape[-1])
    ):
        draft_top_k = int(uniform_top_k_value)
    if (
        isinstance(draft_input, DSparkPPVerifyInputRaw)
        and envs.SGLANG_DSPARK_PP_FAST_REJECTION.get()
        and not sampling_info.is_any_greedy
        and not sampling_info.need_top_p_sampling
        and not getattr(sampling_info, "need_min_p_sampling", False)
    ):
        if not _PP_FAST_REJECTION_LOGGED:
            logger.info(
                "DSpark PP fast rejection is active: top_k=%s max_top_k=%s",
                sampling_info.need_top_k_sampling,
                draft_input.max_top_k,
            )
            _PP_FAST_REJECTION_LOGGED = True
        return accept_sampling_logits_fast(
            candidates=candidates,
            target_logits=target_logits,
            draft_logits=draft_block.corrected_logits,
            temperatures=draft_block.temperatures,
            top_ks=(
                sampling_info.top_ks if sampling_info.need_top_k_sampling else None
            ),
            max_top_k=draft_input.max_top_k,
            uniform_top_k_value=draft_input.uniform_top_k_value,
            draft_top_k=draft_top_k,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )
    bs, gamma_rows, vocab = draft_block.corrected_logits.shape
    if draft_top_k is None:
        draft_probs = SoftmaxTemp.execute(
            logits=draft_block.corrected_logits.reshape(bs * gamma_rows, vocab),
            temperatures=draft_block.temperatures,
            rows_per_request=gamma_rows,
        ).view(bs, gamma_rows, vocab)
    else:
        draft_probs = build_uniform_topk_probs(
            logits=draft_block.corrected_logits,
            temperatures=draft_block.temperatures,
            top_k=draft_top_k,
        )
    if not sampling_info.is_any_greedy:
        return AcceptSampling.execute(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=gamma,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )
    greedy_len, greedy_bonus, greedy_trim = AcceptGreedy.execute(
        candidates=candidates,
        target_logits=target_logits,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_verify_lens=cutoff_verify_lens,
    )
    sampling_len, sampling_bonus, sampling_trim = AcceptSampling.execute(
        candidates=candidates,
        target_logits=target_logits,
        draft_probs=draft_probs,
        sampling_info=sampling_info,
        draft_input=draft_input,
        gamma=gamma,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_verify_lens=cutoff_verify_lens,
    )
    selected = SelectMixedAccept.execute(
        greedy_mask=greedy_mask,
        greedy_len=greedy_len,
        greedy_bonus=greedy_bonus,
        greedy_trim=greedy_trim,
        sampling_len=sampling_len,
        sampling_bonus=sampling_bonus,
        sampling_trim=sampling_trim,
    )
    return selected.correct_len, selected.bonus, selected.cap_trim_lens


def _accept_target_only_draft_tokens(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    sampling_info,
    draft_input: DSparkPPVerifyInputRaw,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verify PP proposal chains when the retained draft distribution is absent."""

    bs = candidates.shape[0]
    if candidates.shape[1] != verify_num_draft_tokens:
        raise ValueError(
            "DSpark target-only candidate width mismatch: "
            f"expected={verify_num_draft_tokens}, got={candidates.shape[1]}."
        )
    if target_logits.shape[0] != bs * verify_num_draft_tokens:
        raise ValueError(
            "DSpark target-only logits row mismatch: "
            f"expected={bs * verify_num_draft_tokens}, "
            f"got={target_logits.shape[0]}."
        )

    if cutoff_verify_lens is None:
        verify_lens = torch.full(
            (bs,),
            verify_num_draft_tokens,
            dtype=torch.int64,
            device=candidates.device,
        )
    else:
        verify_lens = cutoff_verify_lens.to(device=candidates.device, dtype=torch.int64)
        if verify_lens.shape != (bs,):
            raise ValueError(
                "DSpark target-only verify_lens shape mismatch: "
                f"expected={(bs,)}, got={tuple(verify_lens.shape)}."
            )
        if bool(
            torch.any(
                (verify_lens < 1) | (verify_lens > verify_num_draft_tokens)
            ).item()
        ):
            raise ValueError(
                "DSpark target-only verify_lens must be in "
                f"[1, {verify_num_draft_tokens}], got={verify_lens.tolist()}."
            )

    logits_3d = target_logits.view(bs, verify_num_draft_tokens, -1)
    correct_len = torch.empty((bs,), dtype=torch.int32, device=candidates.device)
    bonus = torch.empty((bs,), dtype=torch.int64, device=candidates.device)
    for verify_len_tensor in torch.unique(verify_lens):
        verify_len = int(verify_len_tensor.item())
        indices = torch.nonzero(verify_lens == verify_len, as_tuple=False).view(-1)
        indices_cpu = indices.tolist()
        group_sampling_info = sampling_info
        if indices.numel() != bs:
            group_sampling_info = copy.deepcopy(sampling_info)
            group_sampling_info.filter_batch(indices_cpu, indices)

        group_correct_len, group_bonus = (
            compute_dflash_sampling_correct_drafts_and_bonus(
                candidates=candidates.index_select(0, indices)[
                    :, :verify_len
                ].contiguous(),
                next_token_logits=logits_3d.index_select(0, indices)[:, :verify_len]
                .reshape(-1, target_logits.shape[-1])
                .contiguous(),
                sampling_info=group_sampling_info,
                max_top_k=draft_input.max_top_k,
                uniform_top_k_value=draft_input.uniform_top_k_value,
            )
        )
        correct_len.index_copy_(0, indices, group_correct_len.to(torch.int32))
        bonus.index_copy_(0, indices, group_bonus.to(torch.int64))

    return correct_len, bonus, torch.zeros_like(correct_len)
