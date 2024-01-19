import torch
from torch import nn
import torch.nn.functional as F
from ramp_gae.utils import normalize_abs_sum_to_one, get_sign
from einops import rearrange, repeat


@torch.no_grad()
def subtract_mean(x):
    x_shape = x.shape
    x = x.flatten(1)
    x = x - x.mean(1, keepdim=True)
    x = x.view(x_shape)
    
    return x

@torch.no_grad()
def shift_mean(rel, prel):
    rel_shape, prel_shape = rel.shape, prel.shape
    rel, prel = rel.flatten(1), prel.flatten(1)
    
    shift_val = rel.sum(-1, keepdim=True) - prel.sum(-1, keepdim=True)
    nonzero_mask = (rel != 0).float()
    shift_val = shift_val / nonzero_mask.sum(-1, keepdim=True) * nonzero_mask
    rel = rel - shift_val
    
    rel = rel.view(rel_shape)
    
    return rel

def new_intline_rule(ha, h, x, prel, divide=True, retain_graph=False):
    h_sign = torch.where(h >= 0, 1, -1)
    h_eta = torch.where(h == 0, 1e-9, 0)
    
    prel_scale = prel
    if divide:
        prel_scale = prel_scale * h_sign / (h + h_eta)
    
    rel_pos = torch.autograd.grad(ha + h, x, prel_scale.clamp(min=0), retain_graph=True)[0] * x
    rel_neg = torch.autograd.grad(ha + h, x, prel_scale.clamp(max=0), retain_graph=retain_graph)[0] * x

    prel_pos = prel.flatten(1).clamp(min=0).sum(-1)
    prel_neg = prel.flatten(1).clamp(max=0).sum(-1)
    num_dims = len(rel_pos.shape[1:])
    for _ in range(num_dims):
        prel_pos = prel_pos.unsqueeze(-1)
        prel_neg = prel_neg.unsqueeze(-1)
    
    rel = prel_pos * normalize_abs_sum_to_one(rel_pos) + prel_neg.abs() * normalize_abs_sum_to_one(rel_neg)
    #print('rel', rel_pos.shape, rel_neg.shape, prel_pos.shape, prel_neg.shape, rel.shape)
    return rel

def new_intfill_rule(ha, h, x, prel, divide=True, retain_graph=False):
    h_sign = torch.where(h >= 0, 1, -1)
    h_eta = torch.where(h == 0, 1e-9, 0)
    
    #print(prel.shape, h.shape)
    prel_scale = prel
    if divide:
        prel_scale = prel_scale * h_sign / (h + h_eta)
    
    rel_pos = torch.autograd.grad(ha, x, prel_scale.clamp(min=0), retain_graph=True)[0] * x
    rel_neg = torch.autograd.grad(ha, x, prel_scale.clamp(max=0), retain_graph=retain_graph)[0] * x

    prel_pos = prel.flatten(1).clamp(min=0).sum(-1)
    prel_neg = prel.flatten(1).clamp(max=0).sum(-1)
    num_dims = len(rel_pos.shape[1:])
    for _ in range(num_dims):
        prel_pos = prel_pos.unsqueeze(-1)
        prel_neg = prel_neg.unsqueeze(-1)
    
    rel = prel_pos * normalize_abs_sum_to_one(rel_pos) + prel_neg.abs() * normalize_abs_sum_to_one(rel_neg)
    #print('rel', rel_pos.shape, rel_neg.shape, prel_pos.shape, prel_neg.shape, rel.shape)
    return rel

def new_alpha1beta0_rule(ha, h, x, prel, divide=True, retain_graph=False):
    h_sign = torch.where(h >= 0, 1, -1)
    h_eta = torch.where(h == 0, 1e-9, 0)
    
    #print(prel.shape, h.shape)
    prel_scale = prel
    if divide:
        prel_scale = prel_scale / (ha + h + h_eta)
    
    rel_pos = torch.autograd.grad(ha + h, x, prel_scale.clamp(min=0), retain_graph=True)[0] * x
    rel_neg = torch.autograd.grad(ha + h, x, prel_scale.clamp(max=0), retain_graph=retain_graph)[0] * x

    prel_pos = prel.flatten(1).clamp(min=0).sum(-1)
    prel_neg = prel.flatten(1).clamp(max=0).sum(-1)
    num_dims = len(rel_pos.shape[1:])
    for _ in range(num_dims):
        prel_pos = prel_pos.unsqueeze(-1)
        prel_neg = prel_neg.unsqueeze(-1)
    
    rel = prel_pos * normalize_abs_sum_to_one(rel_pos) + prel_neg.abs() * normalize_abs_sum_to_one(rel_neg)
    #print('rel', rel_pos.shape, rel_neg.shape, prel_pos.shape, prel_neg.shape, rel.shape)
    return rel

class FFModule(nn.Module):
    def backward_prel(self, prel, rule):
        x, h, ha = self.x, self.h, self.ha
        
        h_sign = torch.where(h >= 0, 1, -1)
        prel_sign = torch.where(prel >= 0, 1, -1)
        h_eta = torch.where(h == 0, 1e-9, 0)
        ha_eta = torch.where(ha == 0, 1e-9, 0)
        
        if rule == 'intfill':
            rel = new_intfill_rule(ha, h, x, prel)
        if rule == 'intline':
            rel = new_intline_rule(ha, h, x, prel)
        if rule == 'intneg':
            rel = torch.autograd.grad(ha + h, x, prel * h_sign / (h + h_eta), retain_graph=True)[0] * x - torch.autograd.grad(-ha + h, x, prel * h_sign / (h + h_eta))[0] * x
        if rule == 'intabs':
            rel_fill = torch.autograd.grad(ha, x, prel * h_sign / (h + h_eta), retain_graph=True)[0] * x
            rel_line = torch.autograd.grad(h, x, prel * h_sign / (ha + ha_eta))[0] * x
            rel = normalize_abs_sum_to_one(rel_fill) + normalize_abs_sum_to_one(rel_line)
        if rule == 'intmax':
            rel_fill = torch.autograd.grad(ha, x, prel * h_sign / (h + h_eta), retain_graph=True)[0] * x
            rel_line = torch.autograd.grad(h, x, prel * h_sign)[0] * x
            rel = normalize_abs_sum_to_one(rel_fill) + normalize_abs_sum_to_one(rel_line)
        if rule == 'alpha1beta0':
            rel = new_alpha1beta0_rule(ha, h, x, prel)
        
        self.x, self.h, self.ha = None, None, None
        
        return rel.detach()


class FFLinear(nn.Linear, FFModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        
    @classmethod
    def from_torch(cls, lin):
        bias = lin.bias is not None
        module = cls(lin.in_features, lin.out_features, bias=bias)
        module.load_state_dict(lin.state_dict())

        return module
        
    def forward(self, x):
        #print('linear')
        w, b = self.weight, self.bias
        h = F.linear(x, w, b)
        if b is None:
            ha = F.linear(x.abs(), w.abs())
        else:
            ha = F.linear(x.abs(), w.abs(), b.abs())
        
        xplus, xminus, wplus, wminus = x.clamp(min=0), x.clamp(max=0), w.clamp(min=0), w.clamp(max=0)
        hplus = F.linear(xplus, wplus) + F.linear(xminus, wminus)
        hminus = F.linear(xplus, wminus) + F.linear(xminus, wplus)
            
        self.x, self.h, self.ha, self.hplus, self.hminus = x, h, ha, hplus, hminus
        return h

@torch.no_grad()
def normalize_prel(prel):
    prel = prel.detach().clone()
    prel_shape = prel.shape
    prel = prel.flatten(1)
    prel /= (prel.abs().sum(-1, keepdim=True) + 1e-9)
    prel /= (prel.abs().max(-1, keepdim=True)[0] + 1e-9)
    prel = prel.view(prel_shape)
    
    return prel

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FFSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
        
    def forward(self, x):
        for module in self:
            x = module(x)
        return x
    
    def backward_prel(self, prel, rule):
        for i, module in enumerate(list(self)[::-1]):
            prel = module.backward_prel(prel.detach(), rule)
            #prel = normalize_prel(prel)
        return prel


def double_new_intline_rule(ha, h, x1, x2, prel, divide=True, retain_graph=False, clamp_pos_neg=False):
    h_sign = torch.where(h >= 0, 1, -1)
    h_eta = torch.where(h == 0, 1e-9, 0)
    
    prel_scale = prel * h_sign / (h + h_eta)

    rel_x1_pos = torch.autograd.grad(ha + h, x1, prel_scale.clamp(min=0), retain_graph=True)[0] * x1
    rel_x1_neg = torch.autograd.grad(ha + h, x1, prel_scale.clamp(max=0), retain_graph=True)[0] * x1

    rel_x2_pos = torch.autograd.grad(ha + h, x2, prel_scale.clamp(min=0), retain_graph=True)[0] * x2
    rel_x2_neg = torch.autograd.grad(ha + h, x2, prel_scale.clamp(max=0))[0] * x2

    pos_x1_ratio = rel_x1_pos.flatten(1).sum(-1) / (rel_x1_pos.flatten(1).sum(-1) + rel_x2_pos.flatten(1).sum(-1) + 1e-9)
    pos_x2_ratio = rel_x2_pos.flatten(1).sum(-1) / (rel_x1_pos.flatten(1).sum(-1) + rel_x2_pos.flatten(1).sum(-1) + 1e-9)

    neg_x1_ratio = rel_x1_neg.flatten(1).sum(-1) / (rel_x1_neg.flatten(1).sum(-1) + rel_x2_neg.flatten(1).sum(-1) + 1e-9)
    neg_x2_ratio = rel_x2_neg.flatten(1).sum(-1) / (rel_x1_neg.flatten(1).sum(-1) + rel_x2_neg.flatten(1).sum(-1) + 1e-9)

    prel_pos = prel.flatten(1).clamp(min=0).sum(-1)
    prel_neg = prel.flatten(1).clamp(max=0).sum(-1)
    num_dims = len(rel_x1_pos.shape[1:])
    for _ in range(num_dims):
        prel_pos = prel_pos.unsqueeze(-1)
        prel_neg = prel_neg.unsqueeze(-1)
        pos_x1_ratio = pos_x1_ratio.unsqueeze(-1)
        pos_x2_ratio = pos_x2_ratio.unsqueeze(-1)
        neg_x1_ratio = neg_x1_ratio.unsqueeze(-1)
        neg_x2_ratio = neg_x2_ratio.unsqueeze(-1)

    rel_x1_pos = normalize_abs_sum_to_one(rel_x1_pos)
    rel_x1_neg = normalize_abs_sum_to_one(rel_x1_neg)
    rel_x2_pos = normalize_abs_sum_to_one(rel_x2_pos)
    rel_x2_neg = normalize_abs_sum_to_one(rel_x2_neg)

    rel_x1 = prel_pos * pos_x1_ratio * rel_x1_pos + prel_neg.abs() * neg_x1_ratio * rel_x1_neg
    rel_x2 = prel_pos * pos_x2_ratio * rel_x2_pos + prel_neg.abs() * neg_x2_ratio * rel_x2_neg
    
    return rel_x1, rel_x2


class FFSplitResidual(FFModule):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        #print('resid')
        x1 = x.clone()
        x2 = x.clone()
        
        return x1, x2
    
    def backward_prel(self, prel_x1, prel_x2, rule):
        return (prel_x1 + prel_x2).detach()


class FFCombineResidual(FFModule):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2):
        #print('resid')
        x1, x2 = x1.detach().clone(), x2.detach().clone()
        x1.requires_grad, x2.requires_grad = True, True
        h = x1 + x2
        ha = x1.abs() + x2.abs()
        self.x1, self.x2, self.h, self.ha = x1, x2, h, ha
        
        return h
    
    def backward_prel(self, prel, rule):
        x1, x2, h, ha = self.x1, self.x2, self.h, self.ha
        
        h_sign = torch.where(h >= 0, 1, -1)
        h_eta = torch.where(h == 0, 1e-9, 0)
        ha_eta = torch.where(ha == 0, 1e-9, 0)
        
        if rule == 'intfill':
            #rel_x1 = torch.autograd.grad(ha, x1, prel * h_sign / (h + h_eta), retain_graph=True)[0] * x1
            #rel_x2 = torch.autograd.grad(ha, x2, prel * h_sign / (h + h_eta))[0] * x2
            rel_x1 = torch.autograd.grad(ha + h, x1, prel * h_sign / (h + h_eta), retain_graph=True)[0] * x1
            rel_x2 = torch.autograd.grad(ha + h, x2, prel * h_sign / (h + h_eta))[0] * x2
        elif 'intline' in rule:
            rel_x1, rel_x2 = double_new_intline_rule(ha, h, x1, x2, prel)
        elif rule == 'alpha1beta0':
            rel_x1 = torch.autograd.grad(ha + h, x1, prel / (ha + h + h_eta), retain_graph=True)[0] * x1
            rel_x2 = torch.autograd.grad(ha + h, x2, prel / (ha + h + h_eta))[0] * x2
        
        self.x1, self.x2, self.h, self.ha = [None] * 4
        return rel_x1.detach(), rel_x2.detach()

    
class ExtractHeads(nn.Module):
    def __init__(self, heads):
        super().__init__()
        
        self.heads = heads
        
    def forward(self, x):
        #print('extract')
        h = rearrange(x, 'b n (h d) -> b h n d', h = self.heads)
        ha = rearrange(x, 'b n (h d) -> b h n d', h = self.heads).abs()
        
        self.x, self.h, self.ha = x, h, ha
        
        return h
    
class FFExtractHeads(ExtractHeads, FFModule):
    def __init__(self, heads):
        super().__init__(heads)
    
class SqueezeHeads(nn.Module):
    def __init__(self, heads):
        super().__init__()
        
        self.heads = heads
        
    def forward(self, x):
        #print('squeeze')
        h = rearrange(x, 'b h n d -> b n (h d)')
        ha = rearrange(x, 'b h n d -> b n (h d)').abs()
        
        self.x, self.h, self.ha = x, h, ha
        
        return h
    
class FFSqueezeHeads(SqueezeHeads, FFModule):
    def __init__(self, heads):
        super().__init__(heads)

class FFDynLinear(FFModule):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, w):
        #print('dyn')
        h = torch.matmul(w, x)
        ha = torch.matmul(w.abs(), x.abs())
        
        self.x, self.w, self.h, self.ha = x, w, h, ha
        
        return h
    
    def backward_prel(self, prel, rule):
        x, w, h, ha = self.x, self.w, self.h, self.ha
        
        h_sign = get_sign(h)#torch.where(h >= 0, 1, -1)
        h_eta = torch.where(h == 0, 1e-9, 0)
        
        if rule == 'intfill':
            rel = torch.autograd.grad(ha + h, x, prel * h_sign / (h + h_eta), retain_graph=True)[0] * x
            rel_w = torch.autograd.grad(ha + h, w, prel * h_sign / (h + h_eta))[0] * w
        elif 'intline' in rule:
            rel, rel_w = double_new_intline_rule(ha, h, x, w, prel)
        elif rule == 'alpha1beta0':
            rel = torch.autograd.grad(ha + h, x, prel / (ha + h + h_eta), retain_graph=True)[0] * x
            rel_w = torch.autograd.grad(ha + h, w, prel / (ha + h + h_eta))[0] * w
            
        self.x, self.w, self.h, self.ha = None, None, None, None
            
        return rel.detach(), rel_w.detach()


class FFPatchRearrange(FFModule):
    def __init__(self, nph, npw, ph, pw):
        super().__init__()
        
        self.nph, self.npw, self.ph, self.pw = nph, npw, ph, pw
        
    def forward(self, x):
        #print('patch rearr')
        #return rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.ph, p2=self.pw)
        h = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.ph, p2=self.pw)
        ha = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.ph, p2=self.pw).abs()
        
        self.x, self.h, self.ha = x, h, ha
        
        return h
    

class FFConcat(FFModule):
    def __init__(self, con_token):
        super().__init__()
        
        self.con_token = con_token
        
    def forward(self, x):
        #print('concat')
        b, n, _ = x.shape

        con_tokens = repeat(self.con_token, '1 1 d -> b 1 d', b = b)
        h = torch.cat((con_tokens, x), dim=1)
        ha = torch.cat((con_tokens, x), dim=1).abs()
        
        self.x, self.h, self.ha = x, h, ha
        
        return h

    
class FFChooseToken(FFModule):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        #print('choose')
        self.seq_size = x.shape[1]
        
        h = x[:, 0]
        ha = x[:, 0].abs()
        
        self.x, self.h, self.ha = x, h, ha
        
        return h