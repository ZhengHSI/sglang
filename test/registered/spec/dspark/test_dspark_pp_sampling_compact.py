import os
import types
import unittest
from collections import deque
from contextlib import nullcontext
from unittest import mock

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/flashinfer-test")

import torch

from sglang.srt.models.dspark import run_markov_block
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockResult,
    sample_draft_block,
    sync_dspark_tensor_across_tp,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
    DSparkVerifyPlanner,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DSparkPPVerifyInputRaw,
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
    TargetVerifyResult,
    _sync_accept_across_tp,
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
from sglang.srt.speculative.dspark_components.kernels.dspark_verify_window import (
    scatter_compact_to_strided_into,
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


class TestSchedulerPPOutputSnapshot(CustomTestCase):
    def setUp(self):
        self.scheduler = SchedulerPPMixin()
        self.scheduler.pp_group = mock.Mock()
        self.scheduler.pp_output_group = mock.Mock()

    def test_tensor_message_kinds_use_distinct_pp_channels(self):
        self.scheduler.attn_tp_group = mock.sentinel.attn_tp_group
        self.scheduler.require_attn_tp_allgather = True
        self.scheduler.pp_group.send_tensor_dict.return_value = [
            mock.sentinel.proxy_work
        ]
        self.scheduler.pp_output_group.send_tensor_dict.return_value = [
            mock.sentinel.output_work
        ]
        proxy_payload = {"hidden_states": torch.tensor([1.0])}
        output_payload = {"next_token_ids": torch.tensor([2])}

        proxy_work = self.scheduler._pp_send_dict_to_next_stage(
            proxy_payload,
            msg_type="proxy",
        )
        output_work = self.scheduler._pp_send_dict_to_next_stage(
            output_payload,
            msg_type="output",
        )

        self.assertEqual(proxy_work, [mock.sentinel.proxy_work])
        self.assertEqual(output_work, [mock.sentinel.output_work])
        self.assertNotIn("__msg_type__", proxy_payload)
        self.assertNotIn("__msg_type__", output_payload)
        self.scheduler.pp_group.send_tensor_dict.assert_called_once()
        self.scheduler.pp_output_group.send_tensor_dict.assert_called_once()
        proxy_kwargs = self.scheduler.pp_group.send_tensor_dict.call_args.kwargs
        output_kwargs = (
            self.scheduler.pp_output_group.send_tensor_dict.call_args.kwargs
        )
        self.assertEqual(proxy_kwargs["tensor_dict"]["__msg_type__"], "proxy")
        self.assertEqual(output_kwargs["tensor_dict"]["__msg_type__"], "output")
        self.assertIs(
            proxy_kwargs["all_gather_group"], mock.sentinel.attn_tp_group
        )
        self.assertIsNone(output_kwargs["all_gather_group"])

    def test_tensor_message_kinds_receive_from_distinct_pp_channels(self):
        self.scheduler.pp_group.recv_tensor_dict.return_value = {
            "__msg_type__": "proxy",
            "hidden_states": torch.tensor([1.0]),
        }
        self.scheduler.pp_output_group.recv_tensor_dict.return_value = {
            "__msg_type__": "output",
            "next_token_ids": torch.tensor([2]),
        }

        proxy_payload = self.scheduler._pp_recv_typed_dict(
            expected_kind="proxy",
            all_gather_group=mock.sentinel.attn_tp_group,
        )
        output_payload = self.scheduler._pp_recv_typed_dict(
            expected_kind="output",
            all_gather_group=mock.sentinel.attn_tp_group,
        )

        self.assertEqual(proxy_payload["__msg_type__"], "proxy")
        self.assertEqual(output_payload["__msg_type__"], "output")
        self.scheduler.pp_group.recv_tensor_dict.assert_called_once_with(
            all_gather_group=mock.sentinel.attn_tp_group
        )
        self.scheduler.pp_output_group.recv_tensor_dict.assert_called_once_with(
            all_gather_group=mock.sentinel.attn_tp_group
        )

    def test_output_relay_disables_attn_tp_all_gather(self):
        self.scheduler.require_attn_tp_allgather = True
        self.scheduler.attn_tp_group = mock.sentinel.attn_tp_group
        self.scheduler.pp_output_group.recv_tensor_dict.return_value = {
            "__msg_type__": "output",
            "next_token_ids": torch.tensor([2]),
        }

        payload = self.scheduler._pp_recv_dict_from_prev_stage()

        self.assertEqual(payload["__msg_type__"], "output")
        self.scheduler.pp_output_group.recv_tensor_dict.assert_called_once_with(
            all_gather_group=None
        )

    def test_tensor_channel_kind_mismatch_fails_fast(self):
        self.scheduler.pp_output_group.recv_tensor_dict.return_value = {
            "__msg_type__": "proxy",
            "next_token_ids": torch.tensor([2]),
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "expected 'output', got 'proxy'",
        ):
            self.scheduler._pp_recv_typed_dict(expected_kind="output")

    def test_launch_records_ready_event_after_output_snapshot(self):
        call_order = []
        event = mock.Mock()
        event.record.side_effect = lambda stream: call_order.append("record")
        self.scheduler.device_module = types.SimpleNamespace(
            Event=lambda: event,
            current_stream=lambda: mock.sentinel.current_stream,
        )
        self.scheduler.forward_stream_ctx = nullcontext()
        self.scheduler.forward_stream = mock.Mock()
        self.scheduler.schedule_stream = mock.sentinel.schedule_stream
        self.scheduler.pp_group.is_last_rank = True
        self.scheduler.run_batch = mock.Mock(
            return_value=types.SimpleNamespace(can_run_cuda_graph=False)
        )
        self.scheduler._pp_prepare_tensor_dict = mock.Mock(
            side_effect=lambda result, batch: (
                call_order.append("snapshot") or {"output": torch.tensor([1])}
            )
        )
        mb_metadata = [None]
        last_rank_comm_queue = deque()

        self.scheduler._pp_launch_batch(
            0,
            types.SimpleNamespace(reqs=[]),
            pp_proxy_tensors=None,
            mb_metadata=mb_metadata,
            last_rank_comm_queue=last_rank_comm_queue,
        )

        self.assertEqual(call_order, ["snapshot", "record"])
        self.assertEqual(len(last_rank_comm_queue), 1)

    def test_prepare_snapshots_all_dspark_output_tensors(self):
        self.scheduler.spec_algorithm = types.SimpleNamespace(
            is_dspark=lambda: True
        )
        next_token_ids = torch.tensor([11, 12])
        raw_payload = {
            "dspark_pp_spec_version": 2,
            "dspark_pp_bonus_tokens": torch.tensor([21, 22]),
            "dspark_pp_draft_tokens": torch.tensor([[31, 32], [33, 34]]),
            "dspark_pp_new_seq_lens": torch.tensor([41, 42]),
            "dspark_pp_accept_lens": torch.tensor([1, 2], dtype=torch.int32),
            "dspark_pp_confidence": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            "dspark_pp_cap_trim_lens": torch.tensor([0, 1], dtype=torch.int32),
            "dspark_pp_verify_lens": torch.tensor([3, 3]),
            "dspark_pp_next_verify_lens": torch.tensor([2, 3]),
            "dspark_pp_draft_logits_cache_ids": torch.tensor([7, 7]),
            "dspark_pp_draft_logits_cache_rows": torch.tensor([0, 1]),
            "dspark_pp_max_top_k": 32,
        }
        source_tensors = {
            "next_token_ids": next_token_ids,
            **{
                key: value
                for key, value in raw_payload.items()
                if isinstance(value, torch.Tensor)
            },
        }
        expected = {key: value.clone() for key, value in source_tensors.items()}

        payload = self.scheduler._pp_prepare_tensor_dict(
            types.SimpleNamespace(
                next_token_ids=next_token_ids,
                pp_verify_input_raw=types.SimpleNamespace(
                    to_serializable_dict=lambda: raw_payload
                ),
            ),
            types.SimpleNamespace(return_logprob=False),
        )

        for key, source in source_tensors.items():
            self.assertNotEqual(payload[key].data_ptr(), source.data_ptr())
            source.fill_(-1)
            torch.testing.assert_close(payload[key], expected[key])
        self.assertEqual(payload["dspark_pp_spec_version"], 2)
        self.assertEqual(payload["dspark_pp_max_top_k"], 32)

    def test_prepare_uses_single_metadata_object_for_dspark_raw(self):
        self.scheduler.spec_algorithm = types.SimpleNamespace(
            is_dspark=lambda: True
        )
        next_token_ids = torch.tensor([11, 12])

        payload = self.scheduler._pp_prepare_tensor_dict(
            types.SimpleNamespace(
                next_token_ids=next_token_ids,
                pp_verify_input_raw=_raw(count=2),
            ),
            types.SimpleNamespace(return_logprob=False),
        )

        self.assertEqual(set(payload), {"next_token_ids", "pp_spec_output"})
        self.assertNotEqual(
            payload["next_token_ids"].data_ptr(),
            next_token_ids.data_ptr(),
        )
        self.assertEqual(
            payload["pp_spec_output"]["draft_tokens"],
            [[0, 1], [1, 2]],
        )


class TestDSparkPPVerifyInputRaw(CustomTestCase):
    def test_serializable_round_trip_uses_single_metadata_object(self):
        payload = _raw().to_serializable_dict()
        restored = DSparkPPVerifyInputRaw.from_pp_outputs(
            types.SimpleNamespace(tensors=payload)
        )

        self.assertEqual(list(payload), ["pp_spec_output"])
        self.assertFalse(
            any(
                torch.is_tensor(value)
                for value in payload["pp_spec_output"].values()
            )
        )
        self.assertEqual(restored.bonus_tokens, [100, 101, 102])
        self.assertEqual(restored.draft_tokens, [[0, 1], [1, 2], [2, 3]])
        self.assertEqual(restored.next_verify_lens, [2, 3, 2])

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


class TestDSparkPPBatchResultProcessor(CustomTestCase):
    @unittest.skipUnless(
        torch.cuda.is_available(),
        "requires CUDA for PP accept-lens device regression",
    )
    def test_cuda_accept_lens_keeps_seq_lens_cpu_on_cpu(self):
        metrics = types.SimpleNamespace(
            num_generated_tokens=0,
            forward_ct_decode=0,
            update_spec_metrics=mock.Mock(),
            report_decode_stats=mock.Mock(),
        )
        processor = SchedulerBatchResultProcessor(
            is_generation=True,
            disaggregation_mode=None,
            enable_overlap=True,
            enable_overlap_mlx=False,
            server_args=types.SimpleNamespace(enable_metrics=False),
            model_config=types.SimpleNamespace(think_end_id=None),
            token_to_kv_pool_allocator=types.SimpleNamespace(
                free_group_begin=mock.Mock(),
                free_group_end=mock.Mock(),
            ),
            tree_cache=None,
            hisparse_coordinator=None,
            req_to_token_pool=None,
            decode_offload_manager=None,
            metrics_collector=None,
            metrics_reporter=metrics,
            draft_worker=None,
            model_worker=mock.Mock(),
            logprob_result_processor=None,
            output_streamer=types.SimpleNamespace(stream_output=mock.Mock()),
            abort_request=mock.Mock(),
        )
        raw = DSparkPPVerifyInputRaw.from_pp_outputs(
            types.SimpleNamespace(
                tensors=_raw(start=1, count=1).to_tensor_dict(device="cuda")
            )
        )
        self.assertEqual(raw.accept_lens.device.type, "cuda")
        req = types.SimpleNamespace(finished=lambda: True, is_retracted=False)
        batch = types.SimpleNamespace(
            reqs=[req],
            return_logprob=False,
            spec_algorithm=types.SimpleNamespace(is_none=lambda: False),
            spec_info=raw,
            seq_lens=torch.tensor([10], dtype=torch.int64, device="cuda"),
            seq_lens_cpu=torch.tensor([10], dtype=torch.int64),
            seq_lens_sum=10,
            batch_size=lambda: 1,
        )
        result = types.SimpleNamespace(
            copy_done=None,
            routed_experts_output=None,
            indexer_topk_output=None,
            logits_output=None,
            next_token_ids=None,
            can_run_cuda_graph=False,
            accept_lens=torch.tensor([2], dtype=torch.int32),
            num_correct_drafts=0,
            num_block_accept_tokens=0,
            num_cap_tokens=0,
        )

        with mock.patch.object(
            SchedulerBatchResultProcessor,
            "_normalize_decode_outputs",
            return_value=([[]], None),
        ):
            processor.process_batch_result_decode(batch, result)

        self.assertEqual(batch.seq_lens_cpu.device.type, "cpu")
        torch.testing.assert_close(batch.seq_lens_cpu, torch.tensor([12]))
        torch.testing.assert_close(
            batch.seq_lens, torch.tensor([12], device="cuda")
        )


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


class TestDSparkWorkerPPCompactWiring(CustomTestCase):
    def test_pp_raw_compact_accepts_once_outside_graph(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.device = torch.device("cpu")
        worker.verify_num_draft_tokens = 3
        worker._block_pos_offsets = object()
        worker.model_runner = object()
        worker._target_worker = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(model=object())
        )
        worker.server_args = types.SimpleNamespace(enable_dp_attention=False)
        worker._draft_is_moe = False
        worker._pp_enabled = True
        worker._pp_is_last_rank = True
        worker._simulate_acc_len = 0

        seq_lens = mock.MagicMock()
        seq_lens.__len__.return_value = 1
        sampling_info = types.SimpleNamespace(is_all_greedy=True)
        batch = types.SimpleNamespace(
            spec_info=_raw(count=1),
            forward_mode=types.SimpleNamespace(is_idle=lambda: False),
            seq_lens=seq_lens,
            sampling_info=sampling_info,
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            global_num_tokens=None,
            forward_iter=1,
            reqs=[object()],
            spec_verify_tier_num_tokens=2,
        )

        draft_tokens = torch.tensor([[20, 30]], dtype=torch.int64)
        draft_block = DraftBlockResult(
            draft_tokens=draft_tokens,
            corrected_logits=torch.zeros((1, 2, 4)),
            greedy_mask=torch.ones(1, dtype=torch.bool),
            temperatures=torch.ones(1),
        )
        confidence = torch.ones((1, 2))
        worker._draft_block_from_pp_raw = mock.Mock(
            return_value=(
                torch.tensor([[10]], dtype=torch.int64),
                draft_block,
                draft_tokens,
                confidence,
            )
        )

        layout = types.SimpleNamespace(
            verify_lens=torch.tensor([2], dtype=torch.int32),
            verify_lens_cpu=[2],
        )
        worker._verify_planner = types.SimpleNamespace(
            is_static_mode=False,
            verify_budget_from_lens=mock.Mock(return_value=1),
            layout_from_relayed_verify_lens=mock.Mock(return_value=layout),
            should_run_compact=mock.Mock(return_value=True),
            compute_budget_sync=mock.Mock(return_value=1),
            schedule_layout=mock.Mock(return_value=layout),
            verify_lens_for_pp_relay=mock.Mock(return_value=[2]),
        )

        logits_output = types.SimpleNamespace(
            next_token_logits=torch.zeros((2, 4)),
            hidden_states=torch.zeros((2, 2)),
        )
        target_verify = TargetVerifyResult(
            logits_output=logits_output,
            can_run_cuda_graph=True,
        )
        accept = types.SimpleNamespace(
            correct_len=torch.tensor([1], dtype=torch.int32),
            bonus=torch.tensor([99], dtype=torch.int64),
            cap_trim_lens=torch.tensor([0], dtype=torch.int32),
            commit_lens=torch.tensor([2], dtype=torch.int32),
            new_seq_lens=torch.tensor([7], dtype=torch.int64),
            out_tokens=torch.tensor([[20, 99, 0]], dtype=torch.int64),
        )
        # Even if a future epilogue advertises folded accept, PP raw proposals
        # must never arm it: their graph-owned proposal buffers did not survive
        # the pipeline round trip.
        epilogue = types.SimpleNamespace(folds_accept=True, folds_commit=True)
        worker._verify_executor = types.SimpleNamespace(
            verify_epilogue=epilogue,
            run_compact=mock.Mock(
                return_value=(target_verify, torch.zeros((3, 2)))
            ),
            accept_and_finalize=mock.Mock(return_value=accept),
            commit_hidden=mock.Mock(),
        )

        proposal_next = types.SimpleNamespace(
            confidence=confidence,
            draft_hidden=None,
            draft_block_ids=torch.tensor([[99]], dtype=torch.int64),
            draft_block=draft_block,
            confidence_tap=None,
        )
        worker._proposer = types.SimpleNamespace(
            propose=mock.Mock(return_value=proposal_next)
        )
        worker._draft_context = mock.Mock(return_value=nullcontext())
        worker._cache_pp_draft_logits = mock.Mock(return_value=(None, None))
        worker._dp_verify_tier_num_tokens = mock.Mock(return_value=None)
        worker._observers = mock.MagicMock()
        worker._observers.segment.side_effect = lambda _: nullcontext()

        next_draft_input = types.SimpleNamespace()
        verify_window = object()
        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2."
            "alloc_verify_window",
            return_value=verify_window,
        ), mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2."
            "make_next_draft_input",
            return_value=next_draft_input,
        ), mock.patch.object(
            torch,
            "get_device_module",
            return_value=types.SimpleNamespace(
                current_stream=lambda: mock.sentinel.stream
            ),
        ):
            result = worker._forward_decode(batch, on_publish=None)

        self.assertIsNotNone(result.pp_verify_input_raw)
        compact_call = worker._verify_executor.run_compact.call_args.kwargs
        self.assertFalse(compact_call["inject_gate"])
        self.assertIs(compact_call["verify_window"], verify_window)
        torch.testing.assert_close(
            compact_call["verify_ids_2d"],
            torch.tensor([[10, 20, 30]], dtype=torch.int64),
        )
        self.assertTrue(compact_call["verify_ids_2d"].is_contiguous())
        worker._verify_executor.accept_and_finalize.assert_called_once()
        accept_call = (
            worker._verify_executor.accept_and_finalize.call_args.kwargs
        )
        self.assertFalse(accept_call["folded_accept"])
        worker._verify_executor.commit_hidden.assert_called_once()


class TestDSparkTPSamplingSync(CustomTestCase):
    class _FakeTPGroup:
        world_size = 2

        def __init__(self, authoritative_values):
            self.authoritative_values = list(authoritative_values)
            self.broadcast_count = 0
            self.broadcast_shapes = []

        def broadcast(self, tensor, src=0):
            if src != 0:
                raise AssertionError(f"expected local TP source 0, got {src}")
            self.broadcast_shapes.append(tuple(tensor.shape))
            value = self.authoritative_values[self.broadcast_count]
            tensor.copy_(torch.as_tensor(value, dtype=tensor.dtype).view_as(tensor))
            self.broadcast_count += 1
            return tensor

    class _FakeMarkovHead:
        def __init__(self):
            self.prev_tokens = []

        def apply_step_logits(self, logits, *, token_ids, hidden_states):
            del hidden_states
            self.prev_tokens.append(token_ids.clone())
            return logits

        def sample_block(
            self,
            base_logits,
            *,
            first_prev_tokens,
            hidden_states,
            sampler,
        ):
            return run_markov_block(
                self,
                base_logits,
                first_prev_tokens=first_prev_tokens,
                hidden_states=hidden_states,
                sampler=sampler,
            )

    def test_eager_sync_uses_group_broadcast_with_contiguous_input(self):
        tp_group = types.SimpleNamespace(
            world_size=2,
            pynccl_comm=mock.Mock(),
            broadcast=mock.Mock(),
        )
        non_contiguous = torch.arange(6).view(2, 3).t()
        self.assertFalse(non_contiguous.is_contiguous())

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_draft.get_tp_group",
            return_value=tp_group,
        ), mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_draft."
            "torch.cuda.is_current_stream_capturing",
            return_value=False,
        ):
            result = sync_dspark_tensor_across_tp(non_contiguous)

        self.assertTrue(result.is_contiguous())
        tp_group.broadcast.assert_called_once_with(result, src=0)
        tp_group.pynccl_comm.broadcast.assert_not_called()

    def test_cuda_graph_sync_uses_pynccl_with_contiguous_input(self):
        contiguous = mock.Mock()
        contiguous.device = types.SimpleNamespace(type="cuda")
        tensor = mock.Mock()
        tensor.device = types.SimpleNamespace(type="cuda")
        tensor.contiguous.return_value = contiguous
        pynccl_comm = mock.Mock()
        pynccl_comm.change_state.return_value = nullcontext()
        tp_group = types.SimpleNamespace(
            world_size=2,
            pynccl_comm=pynccl_comm,
            broadcast=mock.Mock(),
        )

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_draft.get_tp_group",
            return_value=tp_group,
        ), mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_draft."
            "torch.cuda.is_current_stream_capturing",
            return_value=True,
        ):
            result = sync_dspark_tensor_across_tp(tensor)

        self.assertIs(result, contiguous)
        tensor.contiguous.assert_called_once_with()
        pynccl_comm.change_state.assert_called_once_with(enable=True)
        pynccl_comm.broadcast.assert_called_once_with(contiguous, src=0)
        tp_group.broadcast.assert_not_called()

    def test_cuda_graph_sync_requires_pynccl(self):
        tensor = mock.Mock()
        tensor.device = types.SimpleNamespace(type="cuda")
        tensor.contiguous.return_value = tensor
        tp_group = types.SimpleNamespace(
            world_size=2,
            pynccl_comm=None,
            broadcast=mock.Mock(),
        )

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_draft.get_tp_group",
            return_value=tp_group,
        ), mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_draft."
            "torch.cuda.is_current_stream_capturing",
            return_value=True,
        ), self.assertRaisesRegex(
            RuntimeError,
            "requires PyNCCL during CUDA graph capture",
        ):
            sync_dspark_tensor_across_tp(tensor)

        tp_group.broadcast.assert_not_called()

    def test_non_greedy_draft_broadcasts_every_markov_step(self):
        tp_group = self._FakeTPGroup(([2], [3], [1]))
        markov_head = self._FakeMarkovHead()
        sampling_info = types.SimpleNamespace(
            is_all_greedy=False,
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            need_min_p_sampling=False,
            temperatures=torch.ones(1),
            top_ks=torch.tensor([2], dtype=torch.int32),
        )

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_draft.get_tp_group",
            return_value=tp_group,
        ), mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_draft."
            "envs.SGLANG_DSPARK_FAST_SAMPLING.get",
            return_value=True,
        ), mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_draft."
            "SampleStepTokens.execute",
            side_effect=lambda **_: torch.tensor([0], dtype=torch.int64),
        ):
            result = sample_draft_block(
                base_logits=torch.zeros((1, 3, 4)),
                anchor_tokens=torch.tensor([0]),
                draft_hidden=torch.zeros((1, 3, 2)),
                sampling_info=sampling_info,
                markov_head=markov_head,
                device=torch.device("cpu"),
            )

        self.assertEqual(tp_group.broadcast_count, 3)
        torch.testing.assert_close(
            result.draft_tokens,
            torch.tensor([[2, 3, 1]]),
        )
        self.assertEqual(
            [tokens.tolist() for tokens in markov_head.prev_tokens],
            [[0], [2], [3]],
        )

    def test_eager_accept_syncs_before_finalize(self):
        tp_group = self._FakeTPGroup(
            (
                [
                    [1],
                    [99],
                    [2],
                ],
            )
        )
        executor = object.__new__(TargetVerifyExecutor)
        executor.gamma = 2
        executor.verify_num_draft_tokens = 3
        executor.verify_epilogue = None
        executor._simulate_acc_len = 0
        local_accept = (
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([7], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int32),
        )

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_draft.get_tp_group",
            return_value=tp_group,
        ), mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "accept_draft_tokens",
            return_value=local_accept,
        ):
            result = executor.accept_and_finalize(
                folded_accept=False,
                bs=1,
                verify_ids_2d=torch.tensor([[10, 20, 30]]),
                target_logits=torch.zeros((3, 4)),
                draft_block=DraftBlockResult(
                    draft_tokens=torch.tensor([[20, 30]]),
                    corrected_logits=torch.zeros((1, 2, 4)),
                    greedy_mask=torch.zeros(1, dtype=torch.bool),
                    temperatures=torch.ones(1),
                ),
                sampling_info=types.SimpleNamespace(is_all_greedy=True),
                draft_input=object(),
                layout=None,
                prefix_lens=torch.tensor([5]),
                draft_tokens=torch.tensor([[20, 30]]),
            )

        self.assertEqual(tp_group.broadcast_count, 1)
        torch.testing.assert_close(
            result.correct_len,
            torch.tensor([1], dtype=torch.int32),
        )
        torch.testing.assert_close(result.bonus, torch.tensor([99]))
        torch.testing.assert_close(
            result.cap_trim_lens,
            torch.tensor([2], dtype=torch.int32),
        )
        torch.testing.assert_close(
            result.commit_lens,
            torch.tensor([2], dtype=torch.int32),
        )
        torch.testing.assert_close(result.new_seq_lens, torch.tensor([7]))
        torch.testing.assert_close(
            result.out_tokens,
            torch.tensor([[20, 99, 0]]),
        )

    def test_persistent_accept_sync_broadcasts_fixed_contiguous_block(self):
        tp_group = self._FakeTPGroup(
            (
                [
                    [1, 2, 0, 0],
                    [99, 98, 0, 0],
                    [2, 1, 0, 0],
                ],
            )
        )
        packed_buf = torch.zeros((3, 4), dtype=torch.int64)

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_draft.get_tp_group",
            return_value=tp_group,
        ):
            correct_len, bonus, cap_trim_lens = _sync_accept_across_tp(
                torch.tensor([0, 0], dtype=torch.int32),
                torch.tensor([7, 8], dtype=torch.int64),
                torch.tensor([0, 0], dtype=torch.int32),
                packed_buf=packed_buf,
            )

        self.assertTrue(packed_buf.is_contiguous())
        self.assertEqual(tp_group.broadcast_shapes, [(3, 4)])
        torch.testing.assert_close(
            correct_len,
            torch.tensor([1, 2], dtype=torch.int32),
        )
        torch.testing.assert_close(bonus, torch.tensor([99, 98]))
        torch.testing.assert_close(
            cap_trim_lens,
            torch.tensor([2, 1], dtype=torch.int32),
        )

    def test_greedy_draft_broadcasts_every_markov_step(self):
        tp_group = self._FakeTPGroup(([1], [0]))
        markov_head = self._FakeMarkovHead()
        base_logits = torch.tensor([[[0.0, 1.0], [2.0, 1.0]]])

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_draft.get_tp_group",
            return_value=tp_group,
        ):
            result = sample_draft_block(
                base_logits=base_logits,
                anchor_tokens=torch.tensor([0]),
                draft_hidden=torch.zeros((1, 2, 2)),
                sampling_info=None,
                markov_head=markov_head,
                device=torch.device("cpu"),
            )

        self.assertEqual(tp_group.broadcast_count, 2)
        self.assertEqual(
            [tokens.tolist() for tokens in markov_head.prev_tokens],
            [[0], [1]],
        )
        torch.testing.assert_close(
            result.draft_tokens,
            torch.tensor([[1, 0]]),
        )

    def test_scatter_into_skip_zero_lens_preserves_inactive_blocks(self):
        compact = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        verify_lens = torch.tensor([2, 0, 1], dtype=torch.int64)
        start = torch.tensor([0, 2, 2], dtype=torch.int64)
        out = torch.full((9, 2), -7.0)

        result = scatter_compact_to_strided_into(
            compact=compact,
            verify_lens=verify_lens,
            out=out,
            stride=3,
            fill_value=0.0,
            start=start,
            skip_zero_lens=True,
        )

        self.assertIs(result, out)
        torch.testing.assert_close(
            out,
            torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [0.0, 0.0],
                    [-7.0, -7.0],
                    [-7.0, -7.0],
                    [-7.0, -7.0],
                    [5.0, 6.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ),
        )

        default_out = torch.full((9, 2), -7.0)
        scatter_compact_to_strided_into(
            compact=compact,
            verify_lens=verify_lens,
            out=default_out,
            stride=3,
            fill_value=0.0,
        )
        torch.testing.assert_close(default_out[3:6], torch.zeros((3, 2)))

    def test_pp_scatter_shares_start_and_skips_zero_lens(self):
        epilogue = DsparkVerifyEpilogue(
            max_bs=2,
            verify_num_draft_tokens=3,
            device=torch.device("cpu"),
            fold_accept=False,
        )
        epilogue.strided_logits = torch.empty((6, 4))
        epilogue.strided_hidden = torch.empty((6, 2))

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "scatter_compact_to_strided_into"
        ) as scatter:
            epilogue._scatter(
                compact_logits=torch.zeros((2, 4)),
                compact_hidden=torch.zeros((2, 2)),
                verify_lens=torch.tensor([2, 0], dtype=torch.int32),
                bs=2,
            )

        self.assertEqual(scatter.call_count, 2)
        logits_call, hidden_call = scatter.call_args_list
        self.assertTrue(logits_call.kwargs["skip_zero_lens"])
        self.assertTrue(hidden_call.kwargs["skip_zero_lens"])
        self.assertIs(logits_call.kwargs["start"], hidden_call.kwargs["start"])
        torch.testing.assert_close(
            logits_call.kwargs["start"],
            torch.tensor([0, 2], dtype=torch.int64),
        )

        folded_epilogue = DsparkVerifyEpilogue(
            max_bs=2,
            verify_num_draft_tokens=3,
            device=torch.device("cpu"),
            fold_accept=True,
        )
        folded_epilogue.strided_logits = torch.empty((6, 4))
        folded_epilogue.strided_hidden = torch.empty((6, 2))
        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "scatter_compact_to_strided_into"
        ) as folded_scatter:
            folded_epilogue._scatter(
                compact_logits=torch.zeros((2, 4)),
                compact_hidden=torch.zeros((2, 2)),
                verify_lens=torch.tensor([2, 0], dtype=torch.int32),
                bs=2,
            )
        self.assertTrue(
            all(
                not call.kwargs["skip_zero_lens"]
                for call in folded_scatter.call_args_list
            )
        )

    def test_epilogue_begin_step_clears_inactive_tail(self):
        epilogue = DsparkVerifyEpilogue(
            max_bs=4,
            verify_num_draft_tokens=3,
            device=torch.device("cpu"),
            fold_accept=False,
        )
        epilogue.begin_step(torch.tensor([3, 2, 1]), armed=True)
        epilogue.begin_step(torch.tensor([2]), armed=False)

        torch.testing.assert_close(
            epilogue.verify_lens_buf,
            torch.tensor([2, 0, 0, 0], dtype=torch.int64),
        )
        torch.testing.assert_close(
            epilogue.inject_gate_buf,
            torch.tensor([0], dtype=torch.int32),
        )

    def test_folded_greedy_epilogue_syncs_before_finalize(self):
        tp_group = self._FakeTPGroup(
            (
                [
                    [1],
                    [99],
                    [0],
                ],
            )
        )
        epilogue = DsparkVerifyEpilogue(
            max_bs=1,
            verify_num_draft_tokens=3,
            device=torch.device("cpu"),
        )
        epilogue.strided_logits = torch.zeros((3, 4))
        epilogue.draft_tokens_buf[:2].copy_(torch.tensor([20, 30]))
        local_accept = (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([7], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int32),
        )
        finalized = types.SimpleNamespace(
            commit_lens=torch.tensor([2], dtype=torch.int32),
            new_seq_lens=torch.tensor([7], dtype=torch.int64),
        )

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_draft.get_tp_group",
            return_value=tp_group,
        ), mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "scatter_compact_to_strided_into",
        ), mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "accept_greedy_triton",
            return_value=local_accept,
        ), mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "finalize_accept_lens_triton",
            return_value=finalized,
        ), mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "BuildOutTokens.execute",
            return_value=torch.tensor([[20, 99, 0]]),
        ):
            commit_lens = epilogue._accept(
                input_ids=torch.tensor([10, 20, 30]),
                seq_lens=torch.tensor([5]),
                verify_lens=torch.tensor([3]),
                bs=1,
            )

        self.assertEqual(tp_group.broadcast_count, 1)
        torch.testing.assert_close(
            commit_lens,
            torch.tensor([2], dtype=torch.int32),
        )
        torch.testing.assert_close(epilogue.correct_len_buf[:1], torch.tensor([1]))
        torch.testing.assert_close(epilogue.bonus_buf[:1], torch.tensor([99]))
        torch.testing.assert_close(
            epilogue.commit_lens_buf[:1],
            torch.tensor([2], dtype=torch.int32),
        )
        torch.testing.assert_close(epilogue.new_seq_lens_buf[:1], torch.tensor([7]))
        torch.testing.assert_close(
            epilogue.out_tokens_buf[:1],
            torch.tensor([[20, 99, 0]]),
        )

    def test_pp_epilogue_records_scatter_without_duplicate_accept(self):
        fused_pool = types.SimpleNamespace(
            set_swa_key_buffer_radix_fused_norm_rope=mock.Mock()
        )
        epilogue = DsparkVerifyEpilogue(
            max_bs=1,
            verify_num_draft_tokens=3,
            device=torch.device("cpu"),
            fold_accept=False,
            commit_ctx=types.SimpleNamespace(resolve_pool=lambda: fused_pool),
        )
        epilogue.strided_logits = torch.empty((3, 4))
        epilogue.strided_hidden = torch.empty((3, 2))
        epilogue._scatter = mock.Mock()
        epilogue._accept = mock.Mock()
        epilogue._commit_inject = mock.Mock()

        epilogue(
            compact_logits=torch.zeros((3, 4)),
            compact_hidden=torch.zeros((3, 2)),
            input_ids=torch.tensor([10, 20, 30]),
            seq_lens=torch.tensor([5]),
            req_pool_indices=torch.tensor([0]),
            bs=1,
        )

        self.assertFalse(epilogue.folds_accept)
        self.assertFalse(epilogue.folds_commit)
        epilogue._scatter.assert_called_once()
        epilogue._accept.assert_not_called()
        epilogue._commit_inject.assert_not_called()


class TestTargetVerifyExecutorFullWidthWindow(CustomTestCase):
    @staticmethod
    def _executor():
        executor = object.__new__(TargetVerifyExecutor)
        executor.verify_num_draft_tokens = 3
        executor.model_runner = object()
        executor.verify_epilogue = None
        executor._run_ragged = mock.Mock(
            return_value=TargetVerifyResult(
                logits_output=None,
                can_run_cuda_graph=False,
            )
        )
        return executor

    @staticmethod
    def _inputs():
        verify_window = types.SimpleNamespace(
            positions_2d=torch.tensor(
                [[10, 11, 12], [20, 21, 22]], dtype=torch.int64
            ),
            verify_cache_loc=torch.tensor(
                [100, 101, 102, 200, 201, 202], dtype=torch.int64
            ),
        )
        verify_ids_2d = torch.tensor(
            [[1, 2, 3], [4, 5, 6]], dtype=torch.int64
        )
        return verify_window, verify_ids_2d

    def test_full_width_reuses_strided_window_without_builder(self):
        executor = self._executor()
        verify_window, verify_ids_2d = self._inputs()
        layout = types.SimpleNamespace(
            verify_lens=torch.tensor([3, 3], dtype=torch.int32),
            verify_lens_cpu=[3, 3],
            total_verify_tokens=6,
            graph_num_tokens=6,
        )

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "BuildRaggedVerifyWindow.execute"
        ) as build_window:
            result, hidden = executor.run_compact(
                batch=object(),
                layout=layout,
                verify_window=verify_window,
                verify_ids_2d=verify_ids_2d,
                draft_block_ids=torch.tensor([[1], [4]], dtype=torch.int64),
                draft_tokens=torch.tensor(
                    [[2, 3], [5, 6]], dtype=torch.int64
                ),
                bs=2,
                device="cpu",
                sampling_info=None,
            )

        self.assertIsNone(hidden)
        self.assertIsNone(result.logits_output)
        build_window.assert_not_called()
        ragged_window = executor._run_ragged.call_args.kwargs["ragged_window"]
        torch.testing.assert_close(
            ragged_window.positions, verify_window.positions_2d.reshape(-1)
        )
        torch.testing.assert_close(
            ragged_window.verify_cache_loc, verify_window.verify_cache_loc
        )
        torch.testing.assert_close(
            ragged_window.verify_ids, verify_ids_2d.reshape(-1)
        )
        self.assertEqual(
            ragged_window.positions.data_ptr(),
            verify_window.positions_2d.data_ptr(),
        )
        self.assertIs(
            ragged_window.verify_cache_loc,
            verify_window.verify_cache_loc,
        )
        self.assertEqual(
            ragged_window.verify_ids.data_ptr(),
            verify_ids_2d.data_ptr(),
        )

    def test_short_lens_in_full_bucket_uses_builder(self):
        executor = self._executor()
        verify_window, verify_ids_2d = self._inputs()
        layout = types.SimpleNamespace(
            verify_lens=torch.tensor([2, 3], dtype=torch.int32),
            verify_lens_cpu=[2, 3],
            total_verify_tokens=5,
            graph_num_tokens=6,
        )
        built_window = object()

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "BuildRaggedVerifyWindow.execute",
            return_value=built_window,
        ) as build_window:
            executor.run_compact(
                batch=object(),
                layout=layout,
                verify_window=verify_window,
                verify_ids_2d=verify_ids_2d,
                draft_block_ids=torch.tensor([[1], [4]], dtype=torch.int64),
                draft_tokens=torch.tensor(
                    [[2, 3], [5, 6]], dtype=torch.int64
                ),
                bs=2,
                device="cpu",
                sampling_info=None,
            )

        build_window.assert_called_once()
        self.assertIs(
            executor._run_ragged.call_args.kwargs["ragged_window"],
            built_window,
        )

    def test_missing_host_metadata_uses_builder(self):
        executor = self._executor()
        verify_window, verify_ids_2d = self._inputs()
        layout = types.SimpleNamespace(
            verify_lens=torch.tensor([3, 3], dtype=torch.int32),
            verify_lens_cpu=None,
            total_verify_tokens=None,
            graph_num_tokens=6,
        )
        built_window = object()

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "BuildRaggedVerifyWindow.execute",
            return_value=built_window,
        ) as build_window:
            executor.run_compact(
                batch=object(),
                layout=layout,
                verify_window=verify_window,
                verify_ids_2d=verify_ids_2d,
                draft_block_ids=torch.tensor([[1], [4]], dtype=torch.int64),
                draft_tokens=torch.tensor(
                    [[2, 3], [5, 6]], dtype=torch.int64
                ),
                bs=2,
                device="cpu",
                sampling_info=None,
            )

        build_window.assert_called_once()
        self.assertIs(
            executor._run_ragged.call_args.kwargs["ragged_window"],
            built_window,
        )


class TestTargetVerifyExecutorRaggedSeqLens(CustomTestCase):
    @staticmethod
    def _run(*, backend_self_adds_seq_lens: bool):
        executor = object.__new__(TargetVerifyExecutor)
        executor.verify_num_draft_tokens = 3
        executor._verify_backend_self_adds_seq_lens = mock.Mock(
            return_value=backend_self_adds_seq_lens
        )

        seq_lens_cpu = torch.tensor([10, 20], dtype=torch.int64)
        batch = types.SimpleNamespace(
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=30,
            out_cache_loc=None,
        )
        verify_lens = mock.Mock()
        verify_lens.cpu.return_value.tolist.return_value = [2, 3]
        layout = types.SimpleNamespace(
            verify_lens=verify_lens,
            verify_lens_cpu=None,
        )
        ragged_window = types.SimpleNamespace(
            verify_ids=torch.tensor([1, 2, 3, 4, 5]),
            positions=torch.tensor([10, 11, 20, 21, 22]),
            verify_cache_loc=torch.tensor([100, 101, 200, 201, 202]),
        )
        result = TargetVerifyResult(
            logits_output=None,
            can_run_cuda_graph=True,
        )
        seen = {}

        def forward_prepared_verify(
            *,
            batch,
            verify_input,
            seq_lens_cpu_backup,
            seq_lens_sum_backup,
            pp_proxy_tensors,
        ):
            seen["seq_lens_cpu"] = batch.seq_lens_cpu.clone()
            seen["seq_lens_sum"] = batch.seq_lens_sum
            seen["verify_input"] = verify_input
            seen["pp_proxy_tensors"] = pp_proxy_tensors
            batch.seq_lens_cpu = seq_lens_cpu_backup
            batch.seq_lens_sum = seq_lens_sum_backup
            return result

        executor._forward_prepared_verify = mock.Mock(
            side_effect=forward_prepared_verify
        )
        proxy = object()
        actual = executor._run_ragged(
            batch=batch,
            layout=layout,
            ragged_window=ragged_window,
            sampling_info=None,
            pp_proxy_tensors=proxy,
        )
        return types.SimpleNamespace(
            actual=actual,
            expected=result,
            executor=executor,
            batch=batch,
            original_seq_lens_cpu=seq_lens_cpu,
            layout=layout,
            ragged_window=ragged_window,
            verify_lens=verify_lens,
            proxy=proxy,
            seen=seen,
        )

    def test_self_adding_backend_skips_verify_lens_cpu_and_cpu_mirror_update(self):
        case = self._run(backend_self_adds_seq_lens=True)

        self.assertIs(case.actual, case.expected)
        case.executor._verify_backend_self_adds_seq_lens.assert_called_once_with()
        case.verify_lens.cpu.assert_not_called()
        torch.testing.assert_close(
            case.seen["seq_lens_cpu"],
            case.original_seq_lens_cpu,
        )
        self.assertEqual(case.seen["seq_lens_sum"], 30)
        self.assertIs(case.batch.seq_lens_cpu, case.original_seq_lens_cpu)
        self.assertEqual(case.batch.seq_lens_sum, 30)
        self.assertIs(
            case.seen["verify_input"].ragged_verify_layout,
            case.layout,
        )
        self.assertIs(case.batch.out_cache_loc, case.ragged_window.verify_cache_loc)
        self.assertIs(case.seen["pp_proxy_tensors"], case.proxy)

    def test_non_self_adding_backend_preserves_cpu_mirror_update(self):
        case = self._run(backend_self_adds_seq_lens=False)

        self.assertIs(case.actual, case.expected)
        case.executor._verify_backend_self_adds_seq_lens.assert_called_once_with()
        case.verify_lens.cpu.assert_called_once_with()
        torch.testing.assert_close(
            case.seen["seq_lens_cpu"],
            torch.tensor([12, 23], dtype=torch.int64),
        )
        self.assertEqual(case.seen["seq_lens_sum"], 35)
        self.assertIs(case.batch.seq_lens_cpu, case.original_seq_lens_cpu)
        self.assertEqual(case.batch.seq_lens_sum, 30)
        self.assertIs(
            case.seen["verify_input"].ragged_verify_layout,
            case.layout,
        )
        self.assertIs(case.batch.out_cache_loc, case.ragged_window.verify_cache_loc)
        self.assertIs(case.seen["pp_proxy_tensors"], case.proxy)


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
        verify_window = types.SimpleNamespace(
            positions_2d=torch.arange(6, dtype=torch.int64).view(1, 6),
            verify_cache_loc=torch.arange(100, 106, dtype=torch.int64),
        )
        verify_ids_2d = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int64)

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "BuildRaggedVerifyWindow.execute"
        ) as build_window:
            result, hidden = executor.run_compact(
                batch=object(),
                layout=types.SimpleNamespace(
                    verify_lens=torch.tensor([6], dtype=torch.int32),
                    verify_lens_cpu=[6],
                    total_verify_tokens=6,
                    graph_num_tokens=6,
                ),
                verify_window=verify_window,
                verify_ids_2d=verify_ids_2d,
                draft_block_ids=torch.tensor([[1]], dtype=torch.int64),
                draft_tokens=torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.int64),
                bs=1,
                device="cpu",
                sampling_info=None,
                pp_proxy_tensors=proxy_in,
            )

        build_window.assert_not_called()
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
