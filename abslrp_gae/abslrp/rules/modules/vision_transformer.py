from torch import nn
from abslrp_gae.abslrp.rules.base import LRPModuleRule
from abslrp_gae.abslrp.rules.primitives.vision_transformer import (
    modified_block_forward,
    block_explain,
    modified_attention_forward,
    attention_explain,
    modified_visiontransformer_forward,
    visiontransformer_explain,
)
from abslrp_gae.abslrp.modified_modules.resnet import ResidualAddition
from abslrp_gae.abslrp.modified_modules.vision_transformer import (
    LayerNormalization,
    QKVLayer,
    QKMultiply,
    SoftmaxAttention,
    AttentionVMultiply,
    CLSTokenPool,
    PositionEmbed,
)
from typing import Optional, Callable


class BlockLRPModuleRule(LRPModuleRule):
    def modified_forward_method(self) -> Optional[Callable]:
        return modified_block_forward

    def modified_explain_method(self) -> Callable:
        return block_explain

    def apply(self, module: nn.Module) -> None:
        module.add_module("residual_addition1", ResidualAddition())
        module.add_module("residual_addition2", ResidualAddition())
        module.add_module("norm_layer1", LayerNormalization(module.norm1.eps, module.norm1.weight, module.norm1.bias))
        module.add_module("norm_layer2", LayerNormalization(module.norm2.eps, module.norm2.weight, module.norm2.bias))
        super().apply(module=module)


class AttentionLRPModuleRule(LRPModuleRule):
    def modified_forward_method(self) -> Optional[Callable]:
        return modified_attention_forward

    def modified_explain_method(self) -> Callable:
        return attention_explain

    def apply(self, module: nn.Module) -> None:
        module.add_module(
            "qkv_layer",
            QKVLayer(
                module.qkv,
                module.num_heads,
                module.head_dim,
            ),
        )
        module.add_module("qk_multiply", QKMultiply())
        module.add_module("softmax_attention", SoftmaxAttention())
        module.add_module("attention_v_multiply", AttentionVMultiply())
        super().apply(module=module)


class VisionTransformerLRPModuleRule(LRPModuleRule):
    def modified_forward_method(self) -> Optional[Callable]:
        return modified_visiontransformer_forward

    def modified_explain_method(self) -> Callable:
        return visiontransformer_explain

    def apply(self, module: nn.Module) -> None:
        module.add_module("cls_pool", CLSTokenPool())
        module.add_module("position_embed", PositionEmbed())
        module.add_module("norm_layer", LayerNormalization(module.norm.eps, module.norm.weight, module.norm.bias))
        super().apply(module=module)
