import torch
import torch.nn.functional as F
from ramp_gae.utils import normalize_abs_sum_to_one, scale_relevance_with_output


def rescale_relevance(r):
    r_shape = r.shape
    r = r.flatten(1)
    r = r - r.flatten(1).min(-1, keepdim=True)[0]
    r = normalize_abs_sum_to_one(r)
    r = r.view(r_shape)
    
    return r

class RelevancyMethod():
    def __init__(self, model, is_torchlrp=False, is_beyond=False, relevancy_type='single_output', **kwargs):
        self.model = model
        self.is_torchlrp = is_torchlrp
        self.is_beyond = is_beyond
        self.relevancy_type = relevancy_type
        self.relevancy_type_dict = {
            'single_output': self.relevancy_method,
            'contrastive': self.contrastive_relevancy_method,
            'fake_contrastive': self.contrastive_relevancy_method,
            'raw': self.raw_relevancy_method,
        }
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.posneg = False
    
    def choose_output(self, chosen_index):
        self.chosen_index = chosen_index
    
    def model_pass(self, x):
        if self.is_torchlrp:
            o = self.model.forward(x, explain=True, rule=self.rule)
        elif self.is_beyond:
            #o = self.model.model(x)
            o = self.model(x)
        else:
            o = self.model(x)
        
        return o

    def relevancy(self, x, scale_with_output=False):
        x = x.to(self.device).detach()
        x.requires_grad = True
        
        o = self.model_pass(x)
        
        r = self.relevancy_type_dict[self.relevancy_type](x, output=o).detach().cpu()
        o = o.detach().cpu()
        
        if self.relevancy_type == 'contrastive':
            o = o.softmax(-1)
        
        if scale_with_output:
            r = scale_relevance_with_output(r, o, self.chosen_index)
        self.posneg = False
        
        return r, o, self.chosen_index
    
    def relevancy_method(self, x, **kwargs):
        raise NotImplementedError('relevancy method needs to be implemented')
    
    def contrastive_relevancy_method(self, x, **kwargs):
        return self.relevancy_method(x, **kwargs)
    
    def raw_relevancy_method(self, x, **kwargs):
        return self.relevancy_method(x, **kwargs)

class IntRelevancyMethod(RelevancyMethod):
    def relevancy_method(self, x, **kwargs):
        o = kwargs['output']
        prel = torch.zeros_like(o, device=o.device)
        prel[torch.arange(o.shape[0]), self.chosen_index.to(self.device)] = 1.
        prel = prel.to(self.device)
        
        r = self.model.backward_prel(prel, self.rule)
        self.model.zero_grad()
        
        return r
    
    def raw_relevancy_method(self, x, **kwargs):
        o = kwargs['output']
        
        prel_pos = torch.zeros_like(o, device=self.device)
        prel_pos[torch.arange(o.shape[0]), self.chosen_index.to(self.device)] = 1.

        r = self.model.backward_prel(prel_pos, self.rule)

        o = self.model(x, p_ind=self.chosen_index.to(self.device))
        
        prel_neg = F.one_hot(self.chosen_index.to(self.device), num_classes=1000) - o.softmax(-1)
        prel_neg[torch.arange(o.shape[0]), self.chosen_index.to(self.device)] = 0.
        
        r_neg = self.model.backward_prel(prel_neg, self.rule)
        
        r = normalize_abs_sum_to_one(r) + normalize_abs_sum_to_one(r_neg)
        
        if self.posneg:
            r = r.relu()
        
        self.model.zero_grad()
        
        return r

    def contrastive_relevancy_method(self, x, **kwargs):
        o = kwargs['output']
        
        if self.contrastive_method == 'difference':
        
            prel_pos = torch.zeros_like(o)
            prel_pos[torch.arange(o.shape[0]), self.chosen_index] = 1.
            prel_neg = torch.ones_like(o)#- o.softmax(-1)
            prel_neg[torch.arange(o.shape[0]), self.chosen_index] = 0.

            prel_pos = prel_pos.to(self.device)
            prel_neg = prel_neg.to(self.device)

            r = self.model.backward_prel(prel_pos, self.rule)

            self.model.zero_grad()

            o = self.model(x)#, p_ind=self.chosen_index.to(self.device))
            #prel_neg = -torch.ones_like(o)
            r_neg = self.model.backward_prel(prel_neg, self.rule)

            #r = normalize_abs_sum_to_one(r) + normalize_abs_sum_to_one(normalize_abs_sum_to_one(r) - normalize_abs_sum_to_one(r_neg))
            r = normalize_abs_sum_to_one(r) - normalize_abs_sum_to_one(r_neg)
        elif self.contrastive_method == 'difference_o':
        
            prel_pos = torch.zeros_like(o)
            prel_pos[torch.arange(o.shape[0]), self.chosen_index] = 1.
            prel_neg = o
            prel_neg[torch.arange(o.shape[0]), self.chosen_index] = 0.

            prel_pos = prel_pos.to(self.device)
            prel_neg = prel_neg.to(self.device)
            #prel_neg = torch.ones_like(o, device=self.device)
            #prel_neg[torch.arange(o.shape[0]), self.chosen_index.to(self.device)] = 0.

            #prel = torch.zeros_like(o, device=o.device)
            #prel[torch.arange(o.shape[0]), self.chosen_index.to(self.device)] = 1.
            #prel = prel.to(self.device)

            r = self.model.backward_prel(prel_pos, self.rule)

            self.model.zero_grad()

            o = self.model(x)#, p_ind=self.chosen_index.to(self.device))
            #prel_neg = -torch.ones_like(o)
            r_neg = self.model.backward_prel(prel_neg, self.rule)

            #r = normalize_abs_sum_to_one(r) + normalize_abs_sum_to_one(normalize_abs_sum_to_one(r) - normalize_abs_sum_to_one(r_neg))
            r = normalize_abs_sum_to_one(r) - normalize_abs_sum_to_one(r_neg)
        
        elif self.contrastive_method == 'difference_softmax':
        
            prel_pos = torch.zeros_like(o)
            prel_pos[torch.arange(o.shape[0]), self.chosen_index] = 1.
            prel_neg = o.softmax(-1)
            prel_neg[torch.arange(o.shape[0]), self.chosen_index] = 0.

            prel_pos = prel_pos.to(self.device)
            prel_neg = prel_neg.to(self.device)
            #prel_neg = torch.ones_like(o, device=self.device)
            #prel_neg[torch.arange(o.shape[0]), self.chosen_index.to(self.device)] = 0.

            #prel = torch.zeros_like(o, device=o.device)
            #prel[torch.arange(o.shape[0]), self.chosen_index.to(self.device)] = 1.
            #prel = prel.to(self.device)

            r = self.model.backward_prel(prel_pos, self.rule)

            self.model.zero_grad()

            o = self.model(x)#, p_ind=self.chosen_index.to(self.device))
            #prel_neg = -torch.ones_like(o)
            r_neg = self.model.backward_prel(prel_neg, self.rule)

            #r = normalize_abs_sum_to_one(r) + normalize_abs_sum_to_one(normalize_abs_sum_to_one(r) - normalize_abs_sum_to_one(r_neg))
            r = normalize_abs_sum_to_one(r) - normalize_abs_sum_to_one(r_neg)
        
        #r = r.abs()
        elif self.contrastive_method == 'single_pass':
            prel_pos = 1.001 * F.one_hot(self.chosen_index.to(self.device), num_classes=o.shape[-1]).to(o.device) - torch.ones_like(o) / 1000
            r = self.model.backward_prel(prel_pos, self.rule)
        
        if self.posneg:
            r = r.relu()
        
        self.model.zero_grad()
        
        return r