# Adapted from AWQ and BlockInt8 implementations for W4 mixed precision quantization

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.nn import Module

from sglang.srt.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.parameter import BlockQuantScaleParameter
from sglang.srt.layers.parameter import GroupQuantScaleParameter, ModelWeightParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import set_weight_attrs
from sglang.srt.layers.quantization.int8_utils import apply_w8a8_block_int8_linear

logger = logging.getLogger(__name__)


class MixedPrecisionW4Config(QuantizationConfig):
    """Config class for W4 Mixed Precision quantization.
    
    This quantization method supports:
    - INT4 weights for expert layers (using TensorRT-LLM unpack)
    - Block INT8 for other layers
    - Mixed precision activation schemes
    """

    def __init__(
        self,
        weight_block_size: List[int] = [128, 128],
        weight_group_size: int = 128,
        activation_scheme: str = "dynamic",
        is_checkpoint_fp8_serialized: bool = False,
        ignored_layers: Optional[List[str]] = None,
        expert_layers: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        self.weight_block_size = weight_block_size
        self.weight_group_size = weight_group_size
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        self.expert_layers = expert_layers or []
        
        # W4 quantization parameters
        self.weight_bits = 4
        self.group_size = weight_group_size
        self.pack_factor = 8 // self.weight_bits  # 2 for INT4 INT4 pack to INT8

    def __repr__(self) -> str:
        return (
            f"MixedPrecisionW4Config(weight_block_size={self.weight_block_size}, "
            f"weight_group_size={self.weight_group_size}, "
            f"activation_scheme={self.activation_scheme}, "
            f"ignored_layers={self.ignored_layers}, "
            f"expert_layers={self.expert_layers})"
        )

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_name(self) -> str:
        return "mixed_precision_w4"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # Requires TensorRT-LLM for INT4 unpack
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_cfg.json",
            "hf_quant_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MixedPrecisionW4Config":
        weight_block_size = cls.get_from_keys_or(
            config, ["weight_block_size"], [128, 128]
        )
        weight_group_size = cls.get_from_keys_or(
            config, ["weight_group_size"], 128
        )
        activation_scheme = cls.get_from_keys_or(
            config, ["activation_scheme"], "dynamic"
        )
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        expert_layers = cls.get_from_keys_or(config, ["expert_layers"], None)
        
        return cls(
            weight_block_size=weight_block_size,
            weight_group_size=weight_group_size,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            expert_layers=expert_layers,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        # Use string comparison to avoid circular import
        layer_class_name = layer.__class__.__name__
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                # print("unquant", layer_class_name, prefix)
                return UnquantizedLinearMethod()
            else:
                # Use block INT8 for non-expert layers
                # print("W8Linear ", layer_class_name, prefix)
                return W4BlockInt8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            # print(prefix, layer.layer_id)
            if "shared_experts" in prefix:
        #     # Lazy import to avoid circular dependency
                # print("W8moe ",layer_class_name,  prefix)
                return W4BlockInt8MoEMethod(self)
            elif  "expert" in prefix:
                # print("W4moe ",layer_class_name,  prefix)
                return W4MoEMethod(self)
        return None


class W4MoEMethod(QuantizeMethodBase):
    """MoE method for W4 quantization.
    Supports INT4 weights for expert layers with TensorRT-LLM unpacking.
    """

    def __init__(self, quant_config):
        self.quant_config = quant_config
    
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Lazy import to avoid circular import
        try:
            from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported
        except ImportError:
            # Fallback for circular import
            class FusedMoeWeightScaleSupported:
                class GROUP:
                    value = "group"

        # Fix: Proper tensor parallel support
        # For tensor parallel, intermediate_size is already per-partition

        # Ensure dimensions are divisible by group_size for proper quantization
        if hidden_size % self.quant_config.group_size != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by group_size {self.quant_config.group_size}"
            )
        if intermediate_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                f"Intermediate size {intermediate_size_per_partition} must be divisible by group_size {self.quant_config.group_size}"
            )
        # group_size = self.quant_config.group_size
        # while intermediate_size_per_partition % group_size or hidden_size % group_size:
        #     group_size = group_size // 2
        #     group_size_div_factor *= 2
        #     assert group_size >= 32
        # layer.group_size = group_size
        # layer.group_size_div_factor = group_size_div_factor

        weight_loader = extra_weight_attrs.get("weight_loader")
        assert "weight_loader" in extra_weight_attrs

        # Fix: Use custom weight loader for INT4 weights
        # if weight_loader is not None:
        #     wrapped_weight_loader = W4MoEMethod.get_weight_loader(layer, weight_loader)
        #     extra_weight_attrs["weight_loader"] = wrapped_weight_loader
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )

        # Fix: Use standard parameter names and dimensions like other MoE implementations
        # INT4 packed weights for w13 (gate_up_proj) - column parallel
        w13_weight = torch.nn.Parameter(
            data=torch.empty(
                num_experts, 
                2 * intermediate_size_per_partition, 
                hidden_size // self.quant_config.pack_factor, 
                dtype=torch.int8  # INT4 packed into INT8
            ),
            requires_grad=False,
        )        
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # INT4 packed weights for w2 (down_proj) - row parallel
        # Shape: [num_experts, hidden_size, intermediate_size_per_partition // pack_factor]
        w2_weight = torch.nn.Parameter(
            data=torch.empty(
                num_experts, 
                hidden_size, 
                intermediate_size_per_partition // self.quant_config.pack_factor, 
                dtype=torch.int8  # INT4 packed into INT8
            ),
            requires_grad=False,
        )        
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Scale parameters for w13 - group quantization
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 
                2 * intermediate_size_per_partition, 
                hidden_size // self.quant_config.group_size, 
                dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        # Scale parameters for w2 - group quantization
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 
                hidden_size, 
                intermediate_size_per_partition // self.quant_config.group_size, 
                dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)

        # Fix: Set proper quantization method for weight loading
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Weights are already processed during loading
        return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        # Lazy import to avoid circular import
        try:
            from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
            from sglang.srt.layers.moe.topk import select_experts
        except ImportError as e:
            raise ImportError(f"MoE dependencies not available: {e}")

        # Unpack INT4 weights using TensorRT-LLM
        # try:
        #     import tensorrt_llm
        #     # print("w13_packed", layer.w13_weight.data.shape)
        #     # print("w2_packed", layer.w2_weight.data.shape)
        #     w13_unpacked_device = layer.w13_weight.data.device
        #     w13_unpacked = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8(
        #         layer.w13_weight.data.cpu()
        #     ).to(w13_unpacked_device)
        #     w2_unpacked_device = layer.w2_weight.data.device
        #     w2_unpacked = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8(
        #         layer.w2_weight.data.cpu()
        #     ).to(w2_unpacked_device)

        #     # print("w13_unpacked", w13_unpacked.shape)
        #     # print("w2_unpacked", w2_unpacked.shape)
        # except ImportError:
        #     raise ImportError(
        #         "TensorRT-LLM is required for W4 MoE quantization. "
        #         "Please install tensorrt_llm."
        #     )

        # # Dequantize weights using scales
        # w13_scale = layer.w13_weight_scale_inv.data
        # w2_scale = layer.w2_weight_scale_inv.data

        # # Fix: Proper scale reshaping for MoE group quantization
        # # if hasattr(w13_scale, 'shape') and isinstance(w13_scale, torch.Tensor) and len(w13_scale.shape) >= 3:
        # #     w13_scale = w13_scale.view(w13_scale.shape[0], w13_scale.shape[1], -1, 1)
        # # if hasattr(w2_scale, 'shape') and isinstance(w2_scale, torch.Tensor) and len(w2_scale.shape) >= 3:
        # #     w2_scale = w2_scale.view(w2_scale.shape[0], w2_scale.shape[1], -1, 1)
        # # Dequantize: weight = int4_weight * scale
        # w13_dequantized = dequantize_int4_to_bf16_3d(w13_unpacked, w13_scale, self.quant_config.group_size)
        # w2_dequantized = dequantize_int4_to_bf16_3d(w2_unpacked, w2_scale, self.quant_config.group_size)
        
        # print("dequantize finish")
        # Expert selection
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
        )

        # Use fused_experts with dequantized weights
        # return fused_experts(
        #     x,
        #     w13_dequantized,
        #     w2_dequantized,
        #     topk_weights=topk_weights,
        #     topk_ids=topk_ids,
        #     inplace=inplace,
        #     activation=activation,
        #     apply_router_weight_on_input=apply_router_weight_on_input,
        #     no_combine=no_combine,
        #     routed_scaling_factor=routed_scaling_factor,
        # )

        weight_bits = self.quant_config.weight_bits

        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=inplace,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_int4_w4a16=weight_bits == 4,
            w1_scale=layer.w13_weight_scale_inv,
            w2_scale=layer.w2_weight_scale_inv,
            block_shape=[0, self.quant_config.group_size],
            no_combine=no_combine,
            routed_scaling_factor=routed_scaling_factor,
        )

def dequantize_int4_to_bf16(quant_tensor, scale_tensor, group_size, original_shape=None):
    """Dequantize int4 tensor back to bf16 with block-wise scaling
    
    Args:
        quant_tensor: Quantized int8 tensor (containing int4 values)
        scale_tensor: Scale tensor from quantization
        original_shape: Original shape before padding (optional, for removing padding)
    
    Returns:
        Dequantized bf16 tensor
    """
    
    # Get current shape
    out_features, total_cols = quant_tensor.shape
    num_groups = total_cols // group_size
    
    # Reshape to grouped format: (out_features, num_groups, group_size)
    blocked_quant = quant_tensor.view(out_features, num_groups, group_size)
    
    # Convert to float for computation
    blocked_quant_float = blocked_quant.float()
    
    # Apply scale to dequantize: multiply by scale for each group
    dequant_tensor = blocked_quant_float * scale_tensor.unsqueeze(-1)
    
    # Reshape back to original format
    dequant_tensor = dequant_tensor.view(out_features, total_cols)
    
    # Remove padding if original shape is provided
    if original_shape is not None:
        dequant_tensor = dequant_tensor[:, :original_shape[1]]
    
    # Convert to bf16
    return dequant_tensor.to(torch.bfloat16)

def dequantize_int4_to_bf16_3d(quant_tensor, scale_tensor, group_size):
    """Dequantize 3D int4 tensor back to bf16 with block-wise scaling
    
    Args:
        quant_tensor: Quantized tensor, shape like [batch, dim1, dim2]
        scale_tensor: Scale tensor, shape like [batch, dim1, dim2//group_size]
        group_size: Size of each quantization group (default: 128)
    
    Returns:
        Dequantized bf16 tensor
    """
    batch_size, dim1, dim2 = quant_tensor.shape
    num_groups = dim2 // group_size
    
    # Reshape to grouped format: [batch, dim1, num_groups, group_size]
    blocked_quant = quant_tensor.view(batch_size, dim1, num_groups, group_size)
    
    # Convert to float for computation
    blocked_quant_float = blocked_quant.float()
    
    # Expand scale tensor to match grouped tensor: [batch, dim1, num_groups, 1]
    expanded_scale = scale_tensor.unsqueeze(-1)  # [batch, dim1, num_groups, 1]
    
    # Apply scale to dequantize
    dequant_tensor = blocked_quant_float * expanded_scale
    
    # Reshape back to original format: [batch, dim1, dim2]
    dequant_tensor = dequant_tensor.view(batch_size, dim1, dim2)
    
    # Convert to bf16
    return dequant_tensor.to(torch.bfloat16)

class W4BlockInt8MoEMethod:
    """MoE method for INT8.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic activation scale.

    Limitations:
    Only support block-wise int8 quantization and int8 checkpoint

    Args:
        quant_config: The quantization config.
    """

    def __new__(cls, *args, **kwargs):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoEMethodBase

        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, quant_config):
        self.quant_config = quant_config
        assert self.quant_config.weight_block_size is not None
        assert self.quant_config.is_checkpoint_int8_serialized

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        if self.quant_config.is_checkpoint_int8_serialized:
            params_dtype = torch.int8
        tp_size = get_tensor_model_parallel_world_size()

        block_n, block_k = (
            self.quant_config.weight_block_size[0],
            self.quant_config.weight_block_size[1],
        )
        # NOTE(HandH1998): To ensure proper alignment of the block-wise quantization scales, the output_size of the weights for both the gate and up layers must be divisible by block_n.
        # Required by column parallel or enabling merged weights
        if intermediate_size % block_n != 0:
            raise ValueError(
                f"The output_size of gate's and up's weight = "
                f"{intermediate_size} is not divisible by "
                f"weight quantization block_n = {block_n}."
            )
        if tp_size > 1:
            # Required by row parallel
            if intermediate_size % block_k != 0:
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * ((intermediate_size + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        assert self.quant_config.activation_scheme == "dynamic"
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:
        # Block quant doesn't need to process weights after loading
        return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
        from sglang.srt.layers.moe.topk import select_experts

        # Expert selection
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
        )

        # Expert fusion with INT8 quantization
        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=inplace,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_int8_w8a8=True,
            w1_scale=(layer.w13_weight_scale_inv),
            w2_scale=(layer.w2_weight_scale_inv),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            block_shape=self.quant_config.weight_block_size,
            no_combine=no_combine,
            routed_scaling_factor=routed_scaling_factor,
        )


class W4BlockInt8LinearMethod(LinearMethodBase):
    """Linear method for W4 non-expert layers using block INT8."""

    def __init__(self, quant_config: MixedPrecisionW4Config):
        self.quant_config = quant_config
        assert self.quant_config.weight_block_size is not None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        block_n, block_k = self.quant_config.weight_block_size
        tp_size = get_tensor_model_parallel_world_size()

        # # Required by row parallel
        # if tp_size > 1 and input_size // input_size_per_partition == tp_size:
        #     if input_size_per_partition % block_k != 0:
        #         raise ValueError(
        #             f"Weight input_size_per_partition = "
        #             f"{input_size_per_partition} is not divisible by "
        #             f"weight quantization block_k = {block_k}."
        #         )
        # # Required by column parallel or enabling merged weights
        # if (tp_size > 1 and output_size // output_size_per_partition == tp_size) or len(
        #     output_partition_sizes
        # ) > 1:
        #     for output_partition_size in output_partition_sizes:
        #         if output_partition_size % block_n != 0:
        #             raise ValueError(
        #                 f"Weight output_partition_size = "
        #                 f"{output_partition_size} is not divisible by "
        #                 f"weight quantization block_n = {block_n}."
        #             )

        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # INT8 weights
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.int8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # Fix: Proper block scale dimension calculation
        num_blocks_n = (output_size_per_partition + block_n - 1) // block_n
        num_blocks_k = (input_size_per_partition + block_k - 1) // block_k
        
        # Block scale parameters
        scale = BlockQuantScaleParameter(
            data=torch.empty(
                num_blocks_n,
                num_blocks_k,
                dtype=torch.float32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale_inv", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Fix: Properly handle parameter conversion with type checking
        # if hasattr(layer, 'weight') and layer.weight is not None:
        #     if hasattr(layer.weight, 'data') and isinstance(layer.weight.data, torch.Tensor):
        #         layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)
        # if hasattr(layer, 'weight_scale_inv') and layer.weight_scale_inv is not None:
        #     if hasattr(layer.weight_scale_inv, 'data') and isinstance(layer.weight_scale_inv.data, torch.Tensor):
        #         layer.weight_scale_inv = torch.nn.Parameter(layer.weight_scale_inv.data, requires_grad=False)
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(
            layer.weight_scale_inv.data, requires_grad=False
        )
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Block INT8 dequantization and linear transformation
        # weight = layer.weight.data
        # scale = layer.weight_scale_inv.data
        
        # # Dequantize weights using block scales
        # block_n, block_k = self.quant_config.weight_block_size
        
        # # Fix: Proper scale reshaping for block quantization with type checking
        # if hasattr(scale, 'shape') and len(scale.shape) == 2:
        #     num_blocks_n, num_blocks_k = scale.shape
        #     scale = scale.view(num_blocks_n, 1, num_blocks_k)
        
        # # Dequantize: weight = int8_weight * scale
        # dequantized_weight = (weight.float() * scale).view(weight.shape)
        
        # # Apply linear transformation
        # out = torch.matmul(x, dequantized_weight.t())
        
        # if bias is not None:
        #     out = out + bias
            
        # return out 

        # Ensure we have proper tensor types - ModelWeightParameter and GroupQuantScaleParameter have .data attribute
        # These parameter types wrap the actual tensor in a .data attribute
        weight_tensor = layer.weight.data
        scale_tensor = layer.weight_scale_inv.data
        
        return apply_w8a8_block_int8_linear(
            input=x,
            weight=weight_tensor,
            block_size=self.quant_config.weight_block_size,
            weight_scale=scale_tensor,
            input_scale=None,
            bias=bias,
        )