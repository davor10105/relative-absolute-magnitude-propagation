import torch


@torch.no_grad()
def normalize_abs_sum_to_one(x):
    original_shape = x.shape
    x = x.flatten(1)
    x /= x.abs().sum(-1, keepdim=True) + 1e-9
    x = x.view(*original_shape).detach()
    
    return x

@torch.no_grad()
def normalize_prel(prel):
    prel = prel.detach().clone()
    prel_shape = prel.shape
    prel = prel.flatten(1)
    prel = prel / (prel.abs().sum(-1, keepdim=True) + 1e-9)
    prel = prel / (prel.abs().max(-1, keepdim=True)[0] + 1e-9)
    prel = prel.view(prel_shape)
    
    return prel

def get_sign(h):
    h_sign = torch.where(h >= 0, 1, -1)
    return h_sign


@torch.no_grad()
def l1_distance(x, y):
    return (x.flatten(1) - y.flatten(1)).abs().sum(-1)


@torch.no_grad()
def l2_distance(x, y):
    return (x.flatten(1) - y.flatten(1)).pow(2).sum(-1).pow(0.5)


@torch.no_grad()
def scale_relevance_with_output(r, o, chosen_index):
    r_shape = r.shape
    r = (o[torch.arange(o.shape[0], device=o.device).unsqueeze(0), chosen_index][0].unsqueeze(-1) * r.flatten(1)).detach()
    r = r.view(r_shape)
    
    return r