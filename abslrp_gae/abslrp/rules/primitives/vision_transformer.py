import torch


def modified_block_forward(self, x: torch.tensor) -> torch.tensor:
    shortcut = x
    x = self.norm_layer1(x)
    x = self.attn(x)
    x = self.ls1(x)
    x = self.drop_path1(x)
    x = self.residual_addition1(x, shortcut)

    shortcut = x
    x = self.norm_layer2(x)
    x = self.mlp(x)
    x = self.ls2(x)
    x = self.drop_path2(x)
    x = self.residual_addition2(x, shortcut)

    return x


def modified_attention_forward(self, x: torch.tensor) -> torch.tensor:
    B, N, C = x.shape
    qkv = self.qkv_layer(x)
    q, k, v = qkv.unbind(0)

    q, k = self.q_norm(q), self.k_norm(k)

    q = q * self.scale
    attn = self.qk_multiply(q, k)
    attn = self.softmax_attention(attn)
    attn = self.attn_drop(attn)
    x = self.attention_v_multiply(attn, v)

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def modified_visiontransformer_forward(self, x: torch.tensor) -> torch.tensor:
    x = self.patch_embed(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat([cls_token, x], 1)
    pos_embed = self.pos_embed.expand(x.shape[0], -1, -1)
    x = self.position_embed(x, pos_embed)
    x = self.patch_drop(x)
    x = self.norm_pre(x)
    x = self.blocks(x)
    x = self.norm_layer(x)
    x = self.cls_pool(x)
    x = self.fc_norm(x)
    x = self.head_drop(x)
    x = self.head(x)

    return x


def block_explain(self, output_relevance: torch.tensor, retain_graph: bool = False) -> torch.tensor:
    output_relevance, shortcut_relevance = self.residual_addition2.explain(output_relevance, retain_graph=True)
    output_relevance = self.drop_path2.explain(output_relevance, retain_graph)
    output_relevance = self.ls2.explain(output_relevance, retain_graph)
    output_relevance = self.mlp.explain(output_relevance, retain_graph)
    output_relevance = self.norm_layer2.explain(output_relevance, retain_graph)
    output_relevance = output_relevance + shortcut_relevance

    output_relevance, shortcut_relevance = self.residual_addition1.explain(output_relevance, retain_graph=True)
    output_relevance = self.drop_path1.explain(output_relevance, retain_graph)
    output_relevance = self.ls1.explain(output_relevance, retain_graph)
    output_relevance = self.attn.explain(output_relevance, retain_graph)
    output_relevance = self.norm_layer1.explain(output_relevance, retain_graph)
    output_relevance = output_relevance + shortcut_relevance

    return output_relevance


def attention_explain(self, output_relevance: torch.tensor, retain_graph: bool = False) -> torch.tensor:
    output_relevance = self.proj_drop.explain(output_relevance, retain_graph)
    output_relevance = self.proj.explain(output_relevance, retain_graph)

    B, N, C = output_relevance.shape
    output_relevance = output_relevance.reshape(B, N, self.num_heads, -1).transpose(2, 1)
    attn_relevance, v_relevance = self.attention_v_multiply.explain(output_relevance, retain_graph=True)
    attn_relevance = self.attn_drop.explain(attn_relevance, retain_graph)
    attn_relevance = self.softmax_attention.explain(attn_relevance, retain_graph)
    q_relevance, k_relevance = self.qk_multiply.explain(attn_relevance, retain_graph)
    q_relevance, k_relevance = self.q_norm.explain(q_relevance, retain_graph), self.k_norm.explain(
        k_relevance, retain_graph
    )
    qkv_relevance = torch.stack([q_relevance, k_relevance, v_relevance], 0).permute(1, 3, 0, 2, 4).reshape(B, N, -1)
    output_relevance = self.qkv_layer.explain(qkv_relevance, retain_graph)

    return output_relevance


def visiontransformer_explain(self, output_relevance: torch.tensor, retain_graph: bool = False) -> torch.tensor:
    output_relevance = self.head.explain(output_relevance, retain_graph)
    output_relevance = self.fc_norm.explain(output_relevance, retain_graph)
    output_relevance = self.head_drop.explain(output_relevance, retain_graph)
    output_relevance = self.cls_pool.explain(output_relevance, retain_graph)
    output_relevance = self.norm_layer.explain(output_relevance, retain_graph)
    output_relevance = self.blocks.explain(output_relevance, retain_graph)
    output_relevance = self.norm_pre.explain(output_relevance, retain_graph)
    output_relevance = self.patch_drop.explain(output_relevance, retain_graph)
    output_relevance, posembed_relevance = self.position_embed.explain(output_relevance, retain_graph=True)
    cls_relevance, output_relevance = output_relevance[:, :1], output_relevance[:, 1:]

    B, N, C = output_relevance.shape
    output_relevance = output_relevance.transpose(1, 2).reshape(B, C, int(N**0.5), int(N**0.5))
    output_relevance = self.patch_embed.explain(output_relevance, retain_graph)

    return output_relevance
