from torch import nn
from abslrp_gae.abslrp.rules.base import LRPModuleRule
from abslrp_gae.abslrp.rules.primitives.resnet import (
    modified_basic_block_forward,
    basic_block_explain,
    modified_bottleneck_forward,
    bottleneck_explain,
)
from abslrp_gae.abslrp.modified_modules.resnet import ResidualAddition
from typing import Optional, Callable


class BasicBlockLRPModuleRule(LRPModuleRule):
    def modified_forward_method(self) -> Optional[Callable]:
        return modified_basic_block_forward

    def modified_explain_method(self) -> Callable:
        return basic_block_explain

    def apply(self, module: nn.Module) -> None:
        module.add_module("residual_addition", ResidualAddition())
        super().apply(module=module)


class BottleneckLRPModuleRule(LRPModuleRule):
    def modified_forward_method(self) -> Optional[Callable]:
        return modified_bottleneck_forward

    def modified_explain_method(self) -> Callable:
        return bottleneck_explain

    def apply(self, module: nn.Module) -> None:
        module.add_module("residual_addition", ResidualAddition())
        super().apply(module=module)
