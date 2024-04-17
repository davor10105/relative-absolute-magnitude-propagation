import math
from ramp_gae.ramp.base import *
from ramp_gae.ramp.timm_modules import *


class HFFeedForward(FFModule):
    def __init__(self, ffn):
        super().__init__()
        
        self.dropout = ffn.dropout
        self.lin1 = TimmLinear(ffn.lin1)
        self.lin2 = TimmLinear(ffn.lin2)
        self.activation = TimmGELU(ffn.activation)
    
    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x
    
    def backward_prel(self, prel, rule):
        prel = self.lin2.backward_prel(prel, rule).detach()
        prel = self.activation.backward_prel(prel, rule).detach()
        prel = self.lin1.backward_prel(prel, rule).detach()
        
        return prel


class HFAttention(FFModule):
    def __init__(self, attention):
        super().__init__()

        self.q_lin = TimmLinear(attention.q_lin)
        self.k_lin = TimmLinear(attention.k_lin)
        self.v_lin = TimmLinear(attention.v_lin)
        self.out_lin = TimmLinear(attention.out_lin)
        
        self.dropout = attention.dropout
        
        self.eheads_q = FFExtractHeads(attention.n_heads)
        self.eheads_k = FFExtractHeads(attention.n_heads)
        self.eheads_v = FFExtractHeads(attention.n_heads)
        self.sheads_v = FFSqueezeHeads(attention.n_heads)
        self.attn_v = FFDynLinear()
        self.attend = TimmSoftmax(-1)
        
        self.attn_mul = TimmAttentionMultiply(1 / math.sqrt(attention.dim // attention.n_heads))
    
    def forward(self, x, attention_mask):
        q, k, v = self.q_lin(x), self.k_lin(x), self.v_lin(x)
        bs, k_length, _ = k.shape
        q, k, v = self.eheads_q(q), self.eheads_k(k), self.eheads_v(v)  # b n (h d) -> b h n d
        
        attn = self.attn_mul(q, k)
        
        mask_reshp = (bs, 1, 1, k_length)
        mask = (attention_mask == 0).view(mask_reshp).expand_as(attn)  # (bs, n_heads, q_length, k_length)
        attn = attn.masked_fill(
            mask, torch.tensor(torch.finfo(attn.dtype).min)
        )
        attn = self.attend(attn)
        attn = self.dropout(attn)
        
        h = self.attn_v(v, attn)
        h = self.sheads_v(h)
        h = self.out_lin(h)
        
        return h
    
    def backward_prel(self, prel, rule):
        prel = self.out_lin.backward_prel(prel.detach(), rule)
        prel = self.sheads_v.backward_prel(prel, rule)
        prel, prel_w = self.attn_v.backward_prel(prel, rule)
        ps, ws = prel.flatten(1).abs().sum(-1), prel_w.flatten(1).abs().sum(-1)
        
        prel_w = self.attend.backward_prel(prel_w, rule)
        prel_q, prel_k = self.attn_mul.backward_prel(prel_w, rule)
        qs, ks = prel_q.flatten(1).abs().sum(-1), prel_k.flatten(1).abs().sum(-1)
        
        prel_q, prel_k = self.eheads_q.backward_prel(prel_q, rule), self.eheads_k.backward_prel(prel_k, rule)
        prel_q, prel_k = self.q_lin.backward_prel(prel_q, rule), self.k_lin.backward_prel(prel_k, rule)
        prel = self.eheads_v.backward_prel(prel, rule)
        prel = self.v_lin.backward_prel(prel, rule)
        
        if 'value_only' in rule:
            prel = prel
        elif 'query_key_only' in rule:
            prel = prel_q + prel_k
        elif 'newintline' in rule:
            prel = prel + prel_q + prel_k
        else:
            prel_qk = qs[:, None, None] * normalize_abs_sum_to_one(prel_q).detach() + ks[:, None, None] * normalize_abs_sum_to_one(prel_k).detach()
            prel = ps[:, None, None] * normalize_abs_sum_to_one(prel).detach() + ws[:, None, None] * normalize_abs_sum_to_one(prel_qk).detach()
        
        return prel
        

class HFTransformerBlock(FFModule):
    def __init__(self, transformer_block):
        super().__init__()
        
        self.attention = HFAttention(transformer_block.attention)
        self.sa_layer_norm = TimmLayerNorm(transformer_block.sa_layer_norm)
        self.ffn = HFFeedForward(transformer_block.ffn)
        self.output_layer_norm = TimmLayerNorm(transformer_block.output_layer_norm)
        
        self.res1_split = FFSplitResidual()
        self.res1_combine = FFCombineResidual()
        self.res2_split = FFSplitResidual()
        self.res2_combine = FFCombineResidual()
    
    def forward(self, x, attention_mask):
        x1, x2 = self.res1_split(x)
        x2 = self.attention(x2, attention_mask)
        x = self.res1_combine(x1, x2)
        x = self.sa_layer_norm(x)
        
        x1, x2 = self.res2_split(x)
        x2 = self.ffn(x2)
        x = self.res2_combine(x1, x2)
        x = self.output_layer_norm(x)
        
        return x
    
    def backward_prel(self, prel, rule):
        prel = self.output_layer_norm.backward_prel(prel, rule)
        prel_x1, prel_x2 = self.res2_combine.backward_prel(prel.detach(), rule)
        px1 = prel_x1.flatten(1).abs().sum(-1)
        px2 = prel_x2.flatten(1).abs().sum(-1)
        prel_x2 = self.ffn.backward_prel(prel_x2, rule)
        
        if 'newintline' in rule:
            prel = prel_x1 + prel_x2
        else:
            prel_x1, prel_x2 = normalize_abs_sum_to_one(prel_x1).detach(), normalize_abs_sum_to_one(prel_x2).detach()
            prel = self.res2_split.backward_prel(prel_x1 * px1[:, None, None], prel_x2 * px2[:, None, None], rule)
        
        prel = self.sa_layer_norm.backward_prel(prel, rule)
        prel_x1, prel_x2 = self.res1_combine.backward_prel(prel, rule)
        px1 = prel_x1.flatten(1).abs().sum(-1)
        px2 = prel_x2.flatten(1).abs().sum(-1)
        prel_x2 = self.attention.backward_prel(prel_x2, rule)
        
        if 'newintline' in rule:
            prel = prel_x1 + prel_x2
        else:
            prel_x1, prel_x2 = normalize_abs_sum_to_one(prel_x1).detach(), normalize_abs_sum_to_one(prel_x2).detach()
            prel = self.res1_split.backward_prel(prel_x1 * px1[:, None, None], prel_x2 * px2[:, None, None], rule)
        
        return prel


class HFModuleList(nn.ModuleList):
    def __init__(self, layer):
        super().__init__(layer)
    
    def forward(self, x, attention_mask):
        for module in self:
            x = module(x, attention_mask)
        return x
    
    def backward_prel(self, prel, rule):
        for module in list(self)[::-1]:
            prel = module.backward_prel(prel, rule)
        
        return prel


class HFTransformer(FFModule):
    def __init__(self, transformer):
        super().__init__()
        
        self.layer = HFModuleList([HFTransformerBlock(l) for l in transformer.layer])
    
    def forward(self, x, attention_mask):
        h = self.layer(x, attention_mask)
        return h
    
    def backward_prel(self, prel, rule):
        prel = self.layer.backward_prel(prel, rule)
        
        return prel
        

class HFDistilBert(FFModule):
    def __init__(self, distilbert):
        super().__init__()
        
        self.embeddings = distilbert.embeddings
        self.transformer = HFTransformer(distilbert.transformer)
    
    def forward(self, input_ids, attention_mask):
        h = self.embeddings(input_ids).detach()
        h.requires_grad = True
        h = self.transformer(h, attention_mask)
        
        return h
    
    def backward_prel(self, prel, rule):
        prel = self.transformer.backward_prel(prel, rule)
        
        return prel