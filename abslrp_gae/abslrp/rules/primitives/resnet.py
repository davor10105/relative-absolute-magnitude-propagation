import torch


def modified_basic_block_forward(self, x: torch.tensor) -> torch.tensor:
    shortcut = x

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.drop_block(x)
    x = self.act1(x)
    x = self.aa(x)

    x = self.conv2(x)
    x = self.bn2(x)

    if self.se is not None:
        x = self.se(x)

    if self.drop_path is not None:
        x = self.drop_path(x)

    if self.downsample is not None:
        shortcut = self.downsample(shortcut)
    x = self.residual_addition(x, shortcut)
    x = self.act2(x)

    return x


def modified_bottleneck_forward(self, x: torch.tensor) -> torch.tensor:
    shortcut = x

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.drop_block(x)
    x = self.act2(x)
    x = self.aa(x)

    x = self.conv3(x)
    x = self.bn3(x)

    if self.se is not None:
        x = self.se(x)

    if self.drop_path is not None:
        x = self.drop_path(x)

    if self.downsample is not None:
        shortcut = self.downsample(shortcut)
    x = self.residual_addition(x, shortcut)
    x = self.act3(x)

    return x


def basic_block_explain(self, output_relevance: torch.tensor, retain_graph: bool = False) -> torch.tensor:
    output_relevance = self.act2.explain(output_relevance, retain_graph)
    output_relevance, shortcut_relevance = self.residual_addition.explain(output_relevance, retain_graph=True)
    if self.downsample is not None:
        shortcut_relevance = self.downsample.explain(shortcut_relevance, retain_graph)
    if self.drop_path is not None:
        output_relevance = self.drop_path.explain(output_relevance, retain_graph)
    if self.se is not None:
        output_relevance = self.se.explain(output_relevance, retain_graph)
    output_relevance = self.bn2.explain(output_relevance, retain_graph)
    output_relevance = self.conv2.explain(output_relevance, retain_graph)
    output_relevance = self.aa.explain(output_relevance, retain_graph)
    output_relevance = self.act1.explain(output_relevance, retain_graph)
    output_relevance = self.drop_block.explain(output_relevance, retain_graph)
    output_relevance = self.bn1.explain(output_relevance, retain_graph)
    output_relevance = self.conv1.explain(output_relevance, retain_graph)

    output_relevance = output_relevance + shortcut_relevance

    return output_relevance


def bottleneck_explain(self, output_relevance: torch.tensor, retain_graph: bool = False) -> torch.tensor:
    output_relevance = self.act3.explain(output_relevance, retain_graph)
    output_relevance, shortcut_relevance = self.residual_addition.explain(output_relevance, retain_graph=True)
    if self.downsample is not None:
        shortcut_relevance = self.downsample.explain(shortcut_relevance, retain_graph)
    if self.drop_path is not None:
        output_relevance = self.drop_path.explain(output_relevance, retain_graph)
    if self.se is not None:
        output_relevance = self.se.explain(output_relevance, retain_graph)
    output_relevance = self.bn3.explain(output_relevance, retain_graph)
    output_relevance = self.conv3.explain(output_relevance, retain_graph)
    output_relevance = self.aa.explain(output_relevance, retain_graph)
    output_relevance = self.act2.explain(output_relevance, retain_graph)
    output_relevance = self.drop_block.explain(output_relevance, retain_graph)
    output_relevance = self.bn2.explain(output_relevance, retain_graph)
    output_relevance = self.conv2.explain(output_relevance, retain_graph)
    output_relevance = self.act1.explain(output_relevance, retain_graph)
    output_relevance = self.bn1.explain(output_relevance, retain_graph)
    output_relevance = self.conv1.explain(output_relevance, retain_graph)

    output_relevance = output_relevance + shortcut_relevance

    return output_relevance
