import os
import types
import unittest
from unittest import mock

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/flashinfer-test")

import torch

from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
    DSparkVerifyPlanner,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DSparkPPVerifyInputRaw,
    TargetVerifyExecutor,
    TargetVerifyResult,
    accept_draft_tokens,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    DSparkWorkerV2,
    _PPDraftLogitsCache,
)
from sglang.srt.speculative.dspark_components.kernels.dspark_accept import (
    accept_sampling_logits_fast,
)
from sglang.srt.speculative.dspark_components.kernels.dspark_draft_model import (
    sample_step_tokens,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _SamplingInfoStub:
    def __init__(self, request_ids):
        self.is_all_greedy = False
        self.request_ids = list(request_ids)

    def filter_batch(self, keep_indices, keep_indices_device):
        del keep_indices_device
        self.request_ids = [self.request_ids[index] for index in keep_indices]


def _raw(start: int = 0, count: int = 3) -> DSparkPPVerifyInputRaw:
    values = list(range(start, start + count))
    return DSparkPPVerifyInputRaw(
        bonus_tokens=[100 + value for value in values],
        draft_tokens=[[value, value + 1] for value in values],
        new_seq_lens=[200 + value for value in values],
        accept_lens=[1 + value % 2 for value in values],
        max_top_k=32,
        uniform_top_k_value=32,
        reserved_seq_lens_cpu=torch.tensor(
            [300 + value for value in values], dtype=torch.int64
        ),
        reserved_seq_lens_sum=sum(300 + value for value in values),
        confidence=[[0.1 * value, 0.1 * value + 0.05] for value in values],
        cap_trim_lens=[value % 2 for value in values],
        verify_lens=[3] * count,
        next_verify_lens=[2 + value % 2 for value in values],
        draft_logits_cache_ids=[7] * count,
        draft_logits_cache_rows=values,
    )


class TestDSparkPPVerifyInputRaw(CustomTestCase):
    def test_round_trip_uses_top_level_tensors(self):
        payload = _raw().to_tensor_dict()
        restored = DSparkPPVerifyInputRaw.from_pp_outputs(
            types.SimpleNamespace(tensors=payload)
        )

        self.assertEqual(payload["dspark_pp_spec_version"], 2)
        self.assertTrue(torch.is_tensor(payload["dspark_pp_bonus_tokens"]))
        self.assertTrue(torch.is_tensor(payload["dspark_pp_draft_tokens"]))
        self.assertNotIn("pp_spec_output", payload)
        self.assertEqual(
            restored.spec_input_type, SpecInputType.DFLASH_PP_VERIFY_INPUT_RAW
        )
        self.assertEqual(restored.max_top_k, 32)
        self.assertEqual(restored.uniform_top_k_value, 32)
        torch.testing.assert_close(restored.next_verify_lens, torch.tensor([2, 3, 2]))
        torch.testing.assert_close(
            restored.draft_logits_cache_rows, torch.tensor([0, 1, 2])
        )

    def test_legacy_payload_remains_readable(self):
        legacy = {
            "pp_spec_output": {
                "bonus_tokens": [11],
                "draft_tokens": [[12, 13]],
                "new_seq_lens": [20],
                "accept_lens": [2],
            }
        }

        restored = DSparkPPVerifyInputRaw.from_pp_outputs(legacy)

        self.assertEqual(restored.bonus_tokens, [11])
        self.assertEqual(restored.draft_tokens, [[12, 13]])
        self.assertTrue(DSparkPPVerifyInputRaw.is_in_tensor_dict(legacy))

    def test_filter_updates_all_per_request_fields(self):
        raw = _raw()

        raw.filter_batch(
            new_indices=torch.tensor([2, 0], dtype=torch.int64),
            new_indices_cpu=[2, 0],
        )

        self.assertEqual(raw.bonus_tokens, [102, 100])
        self.assertEqual(raw.draft_tokens, [[2, 3], [0, 1]])
        self.assertEqual(raw.next_verify_lens, [2, 2])
        self.assertEqual(raw.draft_logits_cache_rows, [2, 0])
        torch.testing.assert_close(raw.reserved_seq_lens_cpu, torch.tensor([302, 300]))
        self.assertEqual(raw.reserved_seq_lens_sum, 602)

    def test_merge_preserves_conservative_top_k_metadata(self):
        lhs = _raw(count=1)
        rhs = _raw(start=1, count=1)
        rhs.max_top_k = 64
        rhs.uniform_top_k_value = 64

        lhs.merge_batch(rhs)

        self.assertEqual(lhs.max_top_k, 64)
        self.assertIsNone(lhs.uniform_top_k_value)
        self.assertEqual(lhs.next_verify_lens, [2, 3])
        self.assertEqual(lhs.draft_logits_cache_rows, [0, 1])

    def test_dummy_seeds_full_verify_plan(self):
        batch = types.SimpleNamespace(
            reqs=[object(), object()],
            input_ids=torch.tensor([11, 22], dtype=torch.int64),
            seq_lens=torch.tensor([101, 202], dtype=torch.int64),
        )

        raw = DSparkPPVerifyInputRaw.build_dummy_for_decode(batch, num_draft=6)

        torch.testing.assert_close(raw.bonus_tokens, torch.tensor([11, 22]))
        torch.testing.assert_close(raw.draft_tokens, torch.tensor([[11] * 5, [22] * 5]))
        torch.testing.assert_close(raw.confidence, torch.zeros((2, 5)))
        torch.testing.assert_close(raw.next_verify_lens, torch.tensor([6, 6]))


class TestPPDraftLogitsCache(CustomTestCase):
    def test_take_reorders_filtered_rows(self):
        cache = _PPDraftLogitsCache(max_rows=8)
        logits = torch.arange(3 * 2 * 5, dtype=torch.float32).view(3, 2, 5)
        tokens = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int64)
        cache_ids, cache_rows = cache.put(corrected_logits=logits, draft_tokens=tokens)

        restored = cache.take(
            cache_ids=cache_ids[[2, 0]],
            cache_rows=cache_rows[[2, 0]],
            expected_draft_tokens=tokens[[2, 0]],
        )

        torch.testing.assert_close(restored, logits[[2, 0]])
        self.assertEqual(cache.hits, 1)

    def test_cache_owns_graph_reused_storage(self):
        cache = _PPDraftLogitsCache(max_rows=4)
        logits = torch.arange(12, dtype=torch.float32).view(2, 2, 3)
        tokens = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        expected_logits = logits.clone()
        expected_tokens = tokens.clone()
        cache_ids, cache_rows = cache.put(corrected_logits=logits, draft_tokens=tokens)
        logits.fill_(-1)
        tokens.fill_(-1)

        restored = cache.take(
            cache_ids=cache_ids,
            cache_rows=cache_rows,
            expected_draft_tokens=expected_tokens,
        )

        torch.testing.assert_close(restored, expected_logits)

    def test_missing_handle_falls_back(self):
        cache = _PPDraftLogitsCache(max_rows=4)

        restored = cache.take(
            cache_ids=torch.tensor([99]),
            cache_rows=torch.tensor([0]),
            expected_draft_tokens=torch.tensor([[1, 2]]),
        )

        self.assertIsNone(restored)
        self.assertEqual(cache.misses, 1)

    def test_capacity_evicts_oldest_entry(self):
        cache = _PPDraftLogitsCache(max_rows=2)
        first_ids, first_rows = cache.put(
            corrected_logits=torch.zeros((2, 2, 3)),
            draft_tokens=torch.zeros((2, 2), dtype=torch.int64),
        )
        cache.put(
            corrected_logits=torch.ones((2, 2, 3)),
            draft_tokens=torch.ones((2, 2), dtype=torch.int64),
        )

        restored = cache.take(
            cache_ids=first_ids,
            cache_rows=first_rows,
            expected_draft_tokens=torch.zeros((2, 2), dtype=torch.int64),
        )

        self.assertIsNone(restored)
        self.assertEqual(cache.evictions, 1)

    def test_worker_restores_cached_logits_for_pp_sampling(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.gamma = 2
        worker.ps = types.SimpleNamespace(tp_rank=1)
        worker._pp_retain_draft_logits = True
        worker._pp_draft_logits_cache = _PPDraftLogitsCache(max_rows=4)
        logits = torch.randn(2, 2, 7)
        tokens = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        cache_ids, cache_rows = worker._pp_draft_logits_cache.put(
            corrected_logits=logits, draft_tokens=tokens
        )
        raw = DSparkPPVerifyInputRaw(
            bonus_tokens=torch.tensor([10, 20]),
            draft_tokens=tokens,
            new_seq_lens=torch.tensor([30, 40]),
            accept_lens=torch.tensor([1, 1]),
            confidence=torch.zeros((2, 2)),
            next_verify_lens=torch.tensor([3, 3]),
            draft_logits_cache_ids=cache_ids,
            draft_logits_cache_rows=cache_rows,
        )
        sampling_info = types.SimpleNamespace(
            is_all_greedy=False,
            temperatures=torch.ones(2),
            top_ks=torch.full((2,), 64),
        )
        batch = types.SimpleNamespace(seq_lens=torch.tensor([30, 40]))

        _, draft_block, restored_tokens, _ = worker._draft_block_from_pp_raw(
            raw, batch, sampling_info
        )

        torch.testing.assert_close(draft_block.corrected_logits, logits)
        torch.testing.assert_close(restored_tokens, tokens)


class TestDSparkPPDynamicVerifyPlan(CustomTestCase):
    @staticmethod
    def _planner() -> DSparkVerifyPlanner:
        planner = object.__new__(DSparkVerifyPlanner)
        planner._ragged_verify_mode = RaggedVerifyMode.COMPACT
        planner._schedule_cfg = DSparkScheduleConfig(gamma=5)
        planner.verify_num_draft_tokens = 6
        planner.model_runner = types.SimpleNamespace(
            decode_cuda_graph_runner=None,
            graph_runner=None,
        )
        planner.server_args = types.SimpleNamespace(pp_size=2)
        return planner

    def test_relayed_plan_reconstructs_compact_layout(self):
        planner = self._planner()

        layout = planner.layout_from_relayed_verify_lens(
            verify_lens=[2, 5, 3],
            expected_bs=3,
            device=torch.device("cpu"),
        )

        self.assertIsNotNone(layout)
        self.assertEqual(layout.verify_lens_cpu, [2, 5, 3])
        self.assertEqual(layout.total_verify_tokens, 10)
        self.assertEqual(planner.verify_budget_from_lens([2, 5, 3]), 7)

    def test_relayed_plan_validates_shape_and_range(self):
        planner = self._planner()

        with self.assertRaisesRegex(ValueError, "wrong batch size"):
            planner.layout_from_relayed_verify_lens(
                verify_lens=[2],
                expected_bs=2,
                device=torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "must be in"):
            planner.layout_from_relayed_verify_lens(
                verify_lens=[0, 7],
                expected_bs=2,
                device=torch.device("cpu"),
            )

    def test_nonlast_executor_initializes_without_confidence_head(self):
        server_args = types.SimpleNamespace(
            speculative_dspark_align_verify_tokens_to_graph_tier=False,
            speculative_dspark_confidence_sts_path=None,
        )

        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_RAGGED_VERIFY_MODE": "compact",
                "SGLANG_PREP_IN_CUDA_GRAPH": "1",
            },
        ), mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_planner."
            "is_dp_attention_enabled",
            return_value=False,
        ):
            planner = DSparkVerifyPlanner(
                draft_model=None,
                gamma=5,
                model_runner=object(),
                device=torch.device("cpu"),
                tp_rank=0,
                server_args=server_args,
                verify_num_draft_tokens=6,
                authoritative_scheduler=False,
            )

        self.assertTrue(planner.is_compact_mode)
        self.assertFalse(planner.carries_confidence)
        self.assertFalse(planner.schedules_verify_budget)

    def test_rank_zero_plan_is_authoritative_across_tp(self):
        planner = self._planner()
        group = types.SimpleNamespace(
            world_size=2,
            rank_in_group=1,
            ranks=[8, 9],
            cpu_group=object(),
        )

        def broadcast_rank_zero_plan(tensor, *, src, group):
            self.assertEqual(src, 8)
            tensor.copy_(torch.tensor([2, 4], dtype=torch.int32))

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_planner.get_tp_group",
            return_value=group,
        ), mock.patch.object(
            torch.distributed,
            "broadcast",
            side_effect=broadcast_rank_zero_plan,
        ):
            synced = planner._sync_pp_verify_lens_across_tp([6, 6])

        self.assertEqual(synced, [2, 4])

    def test_rank_zero_budget_is_authoritative_across_tp(self):
        planner = self._planner()
        planner._sync_verify_budget = True
        planner._budget_sync_tensor = torch.empty(1, dtype=torch.int64)
        group = types.SimpleNamespace(
            world_size=2,
            rank_in_group=1,
            ranks=[8, 9],
            cpu_group=object(),
        )

        def broadcast_rank_zero_budget(tensor, *, src, group):
            self.assertEqual(src, 8)
            tensor.fill_(7)

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_planner.get_tp_group",
            return_value=group,
        ), mock.patch.object(
            torch.distributed,
            "broadcast",
            side_effect=broadcast_rank_zero_budget,
        ):
            synced = planner._sync_budget_across_tp(11)

        self.assertEqual(synced, 7)

    def test_compact_verify_relays_pp_proxy_without_logits(self):
        executor = object.__new__(TargetVerifyExecutor)
        executor.verify_num_draft_tokens = 6
        executor.model_runner = object()
        executor.verify_epilogue = None
        proxy_in = object()
        proxy_out = object()
        executor._run_ragged = mock.Mock(
            return_value=TargetVerifyResult(
                logits_output=None,
                can_run_cuda_graph=False,
                pp_hidden_states_proxy_tensors=proxy_out,
            )
        )

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "BuildRaggedVerifyWindow.execute",
            return_value=object(),
        ):
            result, hidden = executor.run_compact(
                batch=object(),
                layout=types.SimpleNamespace(
                    verify_lens=torch.tensor([2], dtype=torch.int32)
                ),
                draft_block_ids=torch.tensor([[1]], dtype=torch.int64),
                draft_tokens=torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.int64),
                bs=1,
                device="cpu",
                sampling_info=None,
                pp_proxy_tensors=proxy_in,
            )

        self.assertIsNone(hidden)
        self.assertIs(result.pp_hidden_states_proxy_tensors, proxy_out)
        self.assertIs(
            executor._run_ragged.call_args.kwargs["pp_proxy_tensors"], proxy_in
        )


class TestDSparkPPSampling(CustomTestCase):
    def test_fast_rejection_dispatches_for_top_k(self):
        candidates = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        sampling_info = types.SimpleNamespace(
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_k_sampling=True,
            need_top_p_sampling=False,
            need_min_p_sampling=False,
            top_ks=torch.tensor([5], dtype=torch.int32),
        )
        draft_block = DraftBlockResult(
            draft_tokens=candidates[:, 1:],
            corrected_logits=torch.randn(1, 2, 64),
            greedy_mask=torch.zeros(1, dtype=torch.bool),
            temperatures=torch.ones(1),
        )
        expected = (
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([4], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int32),
        )

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "envs.SGLANG_DSPARK_DRAFT_TOPK_SAMPLING.get",
            return_value=True,
        ), mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "envs.SGLANG_DSPARK_PP_FAST_REJECTION.get",
            return_value=True,
        ), mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "accept_sampling_logits_fast",
            return_value=expected,
        ) as fast_rejection:
            result = accept_draft_tokens(
                candidates=candidates,
                target_logits=torch.randn(3, 64),
                draft_block=draft_block,
                sampling_info=sampling_info,
                draft_input=_raw(count=1),
                gamma=2,
                verify_num_draft_tokens=3,
            )

        self.assertIs(result, expected)
        call = fast_rejection.call_args.kwargs
        torch.testing.assert_close(call["top_ks"], sampling_info.top_ks)
        self.assertEqual(call["max_top_k"], 32)
        self.assertEqual(call["draft_top_k"], 32)

    def test_missing_pp_logits_uses_target_only_fallback(self):
        candidates = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
        draft_block = DraftBlockResult(
            draft_tokens=candidates[:, 1:],
            corrected_logits=None,
            greedy_mask=torch.zeros(2, dtype=torch.bool),
            temperatures=torch.ones(2),
        )
        target_only_result = (
            torch.tensor([2, 1], dtype=torch.int32),
            torch.tensor([7, 8], dtype=torch.int64),
        )

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "compute_dflash_sampling_correct_drafts_and_bonus",
            return_value=target_only_result,
        ) as target_only:
            correct_len, bonus, cap_trim_lens = accept_draft_tokens(
                candidates=candidates,
                target_logits=torch.randn(6, 32),
                draft_block=draft_block,
                sampling_info=types.SimpleNamespace(is_all_greedy=False),
                draft_input=_raw(count=2),
                gamma=2,
                verify_num_draft_tokens=3,
            )

        target_only.assert_called_once()
        torch.testing.assert_close(correct_len, target_only_result[0])
        torch.testing.assert_close(bonus, target_only_result[1])
        torch.testing.assert_close(cap_trim_lens, torch.zeros_like(correct_len))

    def test_missing_logits_outside_pp_is_rejected(self):
        candidates = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        draft_block = DraftBlockResult(
            draft_tokens=candidates[:, 1:],
            corrected_logits=None,
            greedy_mask=torch.zeros(1, dtype=torch.bool),
            temperatures=torch.ones(1),
        )

        with self.assertRaisesRegex(RuntimeError, "requires draft logits outside PP"):
            accept_draft_tokens(
                candidates=candidates,
                target_logits=torch.randn(3, 32),
                draft_block=draft_block,
                sampling_info=types.SimpleNamespace(is_all_greedy=False),
                draft_input=types.SimpleNamespace(),
                gamma=2,
                verify_num_draft_tokens=3,
            )

    def test_target_only_groups_mixed_compact_widths(self):
        candidates = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.int64)
        target_logits = torch.randn(9, 32)
        sampling_info = _SamplingInfoStub(["a", "b", "c"])
        draft_block = DraftBlockResult(
            draft_tokens=candidates[:, 1:],
            corrected_logits=None,
            greedy_mask=torch.zeros(3, dtype=torch.bool),
            temperatures=torch.ones(3),
        )
        cutoff_layout = types.SimpleNamespace(
            verify_lens=torch.tensor([3, 2, 3], dtype=torch.int32)
        )
        calls = []

        def target_only_side_effect(**kwargs):
            group_candidates = kwargs["candidates"]
            calls.append(
                (
                    group_candidates.shape[1],
                    kwargs["next_token_logits"].shape[0],
                    list(kwargs["sampling_info"].request_ids),
                )
            )
            return (
                torch.full(
                    (group_candidates.shape[0],),
                    group_candidates.shape[1] - 1,
                    dtype=torch.int32,
                ),
                group_candidates[:, 0] + 100,
            )

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "compute_dflash_sampling_correct_drafts_and_bonus",
            side_effect=target_only_side_effect,
        ):
            correct_len, bonus, trim = accept_draft_tokens(
                candidates=candidates,
                target_logits=target_logits,
                draft_block=draft_block,
                sampling_info=sampling_info,
                draft_input=_raw(count=3),
                gamma=2,
                verify_num_draft_tokens=3,
                cutoff_layout=cutoff_layout,
            )

        self.assertEqual(calls, [(2, 2, ["b"]), (3, 6, ["a", "c"])])
        torch.testing.assert_close(
            correct_len, torch.tensor([2, 1, 2], dtype=torch.int32)
        )
        torch.testing.assert_close(
            bonus, torch.tensor([101, 104, 107], dtype=torch.int64)
        )
        torch.testing.assert_close(trim, torch.zeros_like(correct_len))

    def test_fast_logits_path_matches_reference(self):
        generator = torch.Generator().manual_seed(17)
        bs, gamma, vocab = 3, 4, 11
        width = gamma + 1
        target_logits = torch.randn(bs, width, vocab, generator=generator)
        draft_logits = torch.randn(bs, gamma, vocab, generator=generator)
        temperatures = torch.tensor([0.7, 1.0, 1.4])
        candidates = torch.randint(0, vocab, (bs, width), generator=generator)
        uniform = torch.rand(bs, gamma, generator=generator)
        uniform_final = torch.rand(bs, generator=generator)
        target_probs = torch.softmax(
            target_logits.float() / temperatures[:, None, None], dim=-1
        )
        draft_probs = torch.softmax(
            draft_logits.float() / temperatures[:, None, None], dim=-1
        )
        draft_tokens = candidates[:, 1:]
        candidate_target = (
            target_probs[:, :gamma].gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
        )
        candidate_draft = draft_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
        accepted = uniform * candidate_draft < candidate_target
        expected_len = torch.cumprod(accepted.to(torch.int32), dim=1).sum(dim=1)
        rows = torch.arange(bs)
        selected = expected_len.to(torch.long)
        safe_draft = selected.clamp(max=gamma - 1)
        residual = torch.where(
            (expected_len == gamma)[:, None],
            target_probs[rows, selected],
            (target_probs[rows, selected] - draft_probs[rows, safe_draft]).clamp_min(0),
        )
        residual_sum = residual.sum(dim=-1, keepdim=True)
        expected_bonus = (
            torch.cumsum(residual, dim=-1) <= uniform_final[:, None] * residual_sum
        ).sum(dim=-1)
        expected_bonus.clamp_(max=vocab - 1)

        correct_len, bonus, trim = accept_sampling_logits_fast(
            candidates=candidates,
            target_logits=target_logits.view(bs * width, vocab),
            draft_logits=draft_logits,
            temperatures=temperatures,
            verify_num_draft_tokens=width,
            uniform_samples=uniform,
            uniform_samples_final=uniform_final,
        )

        torch.testing.assert_close(correct_len, expected_len.to(torch.int32))
        torch.testing.assert_close(bonus, expected_bonus.to(torch.int64))
        torch.testing.assert_close(trim, torch.zeros_like(correct_len))

    def test_fast_top_k_path_matches_reference(self):
        generator = torch.Generator().manual_seed(29)
        bs, gamma, vocab = 4, 5, 17
        width = gamma + 1
        target_logits = torch.randn(bs, width, vocab, generator=generator)
        draft_logits = torch.randn(bs, gamma, vocab, generator=generator)
        temperatures = torch.tensor([0.6, 0.9, 1.1, 1.5])
        top_ks = torch.tensor([2, 3, 5, 7], dtype=torch.int32)
        candidates = torch.randint(0, vocab, (bs, width), generator=generator)
        uniform = torch.rand(bs, gamma, generator=generator)
        uniform_final = torch.rand(bs, generator=generator)

        scaled_target = target_logits / temperatures[:, None, None]
        max_top_k = int(top_ks.max().item())
        target_topk_logits, target_topk_ids = torch.topk(
            scaled_target, k=max_top_k, dim=-1
        )
        valid = torch.arange(max_top_k)[None, None, :] < top_ks[:, None, None]
        target_topk_probs = torch.softmax(
            target_topk_logits.masked_fill(~valid, float("-inf")), dim=-1
        )
        target_probs = torch.zeros_like(scaled_target)
        target_probs.scatter_(-1, target_topk_ids, target_topk_probs)
        draft_probs = torch.softmax(draft_logits / temperatures[:, None, None], dim=-1)
        draft_tokens = candidates[:, 1:]
        candidate_target = (
            target_probs[:, :gamma].gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
        )
        candidate_draft = draft_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
        accepted = uniform * candidate_draft < candidate_target
        expected_len = torch.cumprod(accepted.to(torch.int32), dim=1).sum(dim=1)
        rows = torch.arange(bs)
        selected = expected_len.to(torch.long)
        safe_draft = selected.clamp(max=gamma - 1)
        residual = torch.where(
            (expected_len == gamma)[:, None],
            target_probs[rows, selected],
            (target_probs[rows, selected] - draft_probs[rows, safe_draft]).clamp_min(0),
        )
        expected_bonus = (
            torch.cumsum(residual, dim=-1)
            <= uniform_final[:, None] * residual.sum(dim=-1, keepdim=True)
        ).sum(dim=-1)
        expected_bonus.clamp_(max=vocab - 1)

        correct_len, bonus, trim = accept_sampling_logits_fast(
            candidates=candidates,
            target_logits=target_logits.view(bs * width, vocab),
            draft_logits=draft_logits,
            temperatures=temperatures,
            top_ks=top_ks,
            max_top_k=max_top_k,
            verify_num_draft_tokens=width,
            uniform_samples=uniform,
            uniform_samples_final=uniform_final,
        )

        torch.testing.assert_close(correct_len, expected_len.to(torch.int32))
        torch.testing.assert_close(bonus, expected_bonus.to(torch.int64))
        torch.testing.assert_close(trim, torch.zeros_like(correct_len))

    def test_top_k_draft_sampler_never_leaves_support(self):
        logits = torch.tensor([[9.0, 8.0, 1.0, 0.0]])
        sampled = sample_step_tokens(
            step_logits=logits,
            temperatures=torch.ones(1),
            greedy_mask=torch.zeros(1, dtype=torch.bool),
            exp_noise=torch.tensor([[1000.0, 1.0]]),
            top_k=2,
        )

        self.assertIn(sampled.item(), (0, 1))


if __name__ == "__main__":
    unittest.main()
