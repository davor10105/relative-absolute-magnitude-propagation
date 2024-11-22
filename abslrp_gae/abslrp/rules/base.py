import torch
from torch import nn
from abslrp_gae.utils import normalize_relevance, NormalizationType
from types import MethodType
from abc import abstractmethod, ABC
from typing import Callable, Type, Optional, Dict
from copy import deepcopy
from pydantic import BaseModel, ConfigDict


class LRPModuleRule(ABC):
    @abstractmethod
    def modified_forward_method(self) -> Optional[Callable]:
        raise NotImplementedError()

    @abstractmethod
    def modified_explain_method(self) -> Callable:
        raise NotImplementedError()

    def apply(self, module: nn.Module) -> None:
        modified_forward_method = self.modified_forward_method()
        if modified_forward_method is not None:
            module.forward = MethodType(modified_forward_method, module)
        module.explain = MethodType(self.modified_explain_method(), module)


class AbsLRPModuleRule(LRPModuleRule):
    def __init__(self, divide_relevance=True):
        self.divide_relevance = divide_relevance

    def modified_forward_method(self) -> Optional[Callable]:
        return None

    def modified_explain_method(self) -> Callable:
        return construct_abslrp_rule(self.divide_relevance)

    def apply(self, module: nn.Module) -> None:
        module.register_forward_hook(abslrp_forward_hook)
        super().apply(module=module)


class SequentialAbsLRPModuleRule(LRPModuleRule):
    def modified_forward_method(self) -> Optional[Callable]:
        return None

    def modified_explain_method(self) -> Callable:
        return sequential_abslrp_rule


class IdentityLRPModuleRule(LRPModuleRule):
    def modified_forward_method(self) -> Optional[Callable]:
        return None

    def modified_explain_method(self) -> Callable:
        return identity_rule


class LRPRule(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    module_to_rule_dict: Dict[Type[nn.Module], LRPModuleRule] = {}
    default_rule: LRPModuleRule
    default_composite_rule: LRPModuleRule

    def apply(self, module: nn.Module) -> None:
        # apply special rule if exists
        rule_applied = False
        for module_class, rule in self.module_to_rule_dict.items():
            if isinstance(module, module_class):
                rule.apply(module=module)
                rule_applied = True
                break
        # otherwise, apply default rule
        if not rule_applied:
            if not list(module.children()):
                self.default_rule.apply(module=module)
            else:
                self.default_composite_rule.apply(module=module)
        # apply this rule to all children of the module
        for child_module in module.children():
            self.apply(child_module)


class AbsLRPRule(LRPRule):
    default_rule: LRPModuleRule = AbsLRPModuleRule()
    default_composite_rule: LRPModuleRule = SequentialAbsLRPModuleRule()


def identity_rule(self, output_relevance: torch.tensor, retain_graph: bool = False) -> torch.tensor:
    return output_relevance


def construct_abslrp_rule(divide_relevance: bool) -> Callable:
    def abslrp_rule(self, output_relevance: torch.tensor, retain_graph: bool = False) -> torch.tensor:
        input = self.saved_tensors["input"]
        output = self.saved_tensors["output"]
        if "abs_output" in self.saved_tensors:
            output = output + self.saved_tensors["abs_output"]
        if divide_relevance:
            output_relevance = output_relevance / (self.saved_tensors["output"].abs() + 1e-9)
        # apply abslrp rule to each input
        grads = torch.autograd.grad(output, input, output_relevance, retain_graph=retain_graph)
        relevances = [arg * grad for arg, grad in zip(input, grads)]
        # normalize relevances
        relevances = [
            normalize_relevance(relevance=relevance, normalization_type=NormalizationType.SUM_TO_ONE)
            for relevance in relevances
        ]
        # delete saved tensors if not needed
        if not retain_graph:
            self.saved_tensors = {}
        if len(relevances) == 1:
            return relevances[0]
        return relevances

    return abslrp_rule


def sequential_abslrp_rule(self, output_relevance: torch.tensor, retain_graph: bool = False) -> torch.tensor:
    for child in list(self.children())[::-1]:
        output_relevance = child.explain(output_relevance=output_relevance, retain_graph=retain_graph)
    return output_relevance


def abslrp_forward_hook(module: nn.Module, args: tuple, output: torch.tensor) -> None:
    module.saved_tensors = {}
    # create a temporary temp module
    abs_module = deepcopy(module)
    # remove this hook from copied model
    if getattr(abs_module, "_forward_hooks", None):
        for i, hook in abs_module._forward_hooks.items():
            if hook.__name__ == "abslrp_forward_hook":
                break
        del abs_module._forward_hooks[i]
    # if module has learnable parameters, infer over absolute parameters
    if getattr(module, "weight", None) is not None:
        abs_module.weight.data = abs_module.weight.data.abs()
        if getattr(module, "bias", None) is not None:
            abs_module.bias.data = abs_module.bias.data.abs()
    abs_output = abs_module(*[arg.abs() for arg in args])
    # save the outputs and inputs
    module.saved_tensors["abs_output"] = abs_output
    module.saved_tensors["output"] = output
    module.saved_tensors["input"] = args
