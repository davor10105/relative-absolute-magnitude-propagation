from torch import nn
from timm.models.resnet import BasicBlock, Bottleneck
from timm.models.vision_transformer import Block, Attention, VisionTransformer
from abslrp_gae.abslrp.rules.base import AbsLRPRule, AbsLRPModuleRule, LRPModuleRule, IdentityLRPModuleRule
from abslrp_gae.abslrp.rules.modules.resnet import BasicBlockLRPModuleRule, BottleneckLRPModuleRule
from abslrp_gae.abslrp.rules.modules.vision_transformer import (
    BlockLRPModuleRule,
    AttentionLRPModuleRule,
    VisionTransformerLRPModuleRule,
)
from typing import Dict, Type


class VGGAbsLRPRule(AbsLRPRule):
    module_to_rule_dict: Dict[Type[nn.Module], LRPModuleRule] = {nn.MaxPool2d: AbsLRPModuleRule(divide_relevance=False)}


class ResNetAbsLRPRule(AbsLRPRule):
    module_to_rule_dict: Dict[Type[nn.Module], LRPModuleRule] = {
        BasicBlock: BasicBlockLRPModuleRule(),
        Bottleneck: BottleneckLRPModuleRule(),
    }


class VisionTransformerAbsLRPRule(AbsLRPRule):
    module_to_rule_dict: Dict[Type[nn.Module], LRPModuleRule] = {
        Block: BlockLRPModuleRule(),
        Attention: AttentionLRPModuleRule(),
        VisionTransformer: VisionTransformerLRPModuleRule(),
        nn.Identity: IdentityLRPModuleRule(),
    }
