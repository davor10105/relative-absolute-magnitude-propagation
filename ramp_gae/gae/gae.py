import torch
import torch.nn.functional as F
from tqdm import tqdm
from ramp_gae.utils import *
import matplotlib.pyplot as plt
from ramp_gae.ramp.relevancy_methods import RelevancyMethod
from typing import Dict


def mask_reversemask_with_relevance(masking_input, masking_scoring, percentage, mask_top):
    masking_input, masking_scoring = masking_input.detach().cpu(), masking_scoring.detach().cpu()
    
    original_shape = masking_input.shape
    masking_input = masking_input.flatten(1)
    masking_scoring = masking_scoring.flatten(1)
    
    top_ret = masking_scoring.topk(k=int(percentage * masking_scoring.size(1)), largest=mask_top)

    #mask_x = masking_f(masking_input, device=masking_input.device)
    rev_mask_x = masking_input.clone().detach()
    one_zero_mask = torch.ones_like(masking_input)
    #mask_x.scatter_(1, top_ret.indices, masking_input[torch.arange(masking_input.size(0)).unsqueeze(1), top_ret.indices])
    rev_mask_x.scatter_(1, top_ret.indices, torch.zeros_like(masking_input)[torch.arange(masking_input.size(0)).unsqueeze(1), top_ret.indices])
    one_zero_mask.scatter_(1, top_ret.indices, torch.zeros_like(masking_input)[torch.arange(masking_input.size(0)).unsqueeze(1), top_ret.indices])
    #mask_x = mask_x.view(*original_shape)
    rev_mask_x = rev_mask_x.view(*original_shape)
    one_zero_mask = one_zero_mask.view(*original_shape)

    return rev_mask_x.detach(), one_zero_mask.detach()


def foreground_background_adversarial_attack_for_stepwise_comparison(x_original, model, target, mode, fill_f=torch.clone, masking_f=torch.zeros_like, steps=10, num_classes=1000):
    assert mode in ['foreground', 'background']
    
    x = x_original.clone().detach()
    x.requires_grad = True
    #x = torch.stack([imagenet_test_data[image_index] for image_index in image_indices]).to(device)
    #x.requires_grad = True
    
    x_start = x.detach().clone().cpu()
    all_xs = [x.detach().clone().cpu()]
    all_os = []
    one_zero_masks = []
    
    x_diffs = torch.zeros_like(x).cpu()
    
    b = x.shape[0]
    chosen_inverse_index = torch.arange(num_classes)[None, :].repeat(b, 1)
    inverse_mask = chosen_inverse_index != target[:, None].cpu()
    
    step_size = 1 / steps
    for percentage in torch.linspace(step_size, 1 - step_size, steps):
        model.zero_grad()

        o = model(x)
        
        loss = o[torch.arange(o.shape[0]), target].abs().mean()
        #loss = nn.CrossEntropyLoss()(o, target.to(o.device))
        loss.backward()

        inputgrad_r = (x.grad.data * x.data).abs()
        
        x, one_zero_mask = mask_reversemask_with_relevance(x, inputgrad_r, percentage, mask_top=mode=='foreground')
        
        x = x.to(o.device)
        x.requires_grad = True
        
        #x_diffs = x_diffs + normalize_prel(inputgrad_r.detach().cpu()) * percentage
        #x_diffs = x_diffs + normalize_prel(inputgrad_r.detach().cpu()) * (1 - percentage)
        x_diffs = x_diffs + inputgrad_r.detach().cpu()# * (1 - percentage)
        
        all_xs.append(x.detach().clone().cpu())
        all_os.append(o[torch.arange(b), target].detach().clone().cpu())
        one_zero_masks.append(one_zero_mask.detach().clone().cpu())
    o = model(x)
    all_os.append(o[torch.arange(b), target].detach().clone().cpu())
        
    all_xs = torch.stack(all_xs, 0)
    #all_os = torch.stack(all_os[1:], 1)
    all_os = torch.stack(all_os, 1)
    one_zero_masks = torch.stack(one_zero_masks, 0)
    x_diffs = normalize_prel(x_diffs)#.abs()
    
    return (all_xs, all_os), x_diffs, one_zero_masks


def combine_insert_remove(x_insert, x_remove):
    return normalize_prel(normalize_abs_sum_to_one(x_insert) - normalize_abs_sum_to_one(x_remove))


class GlobalEvaluationMetric():
    def __init__(self, model, relevancy_methods, test_loader, device='cuda'):
        self.model = model
        self.relevancy_methods: Dict[str, RelevancyMethod] = relevancy_methods
        self.test_loader = test_loader
        self.device = device
    
    def run(self, num_examples=256, step_num=10):
        norm_function = normalize_prel

        final_scores = {}
        detailed_scores = {}
        for example_num, (x_whole, x_indices) in tqdm(enumerate(self.test_loader)):
            x_whole = x_whole.to(self.device)
            
            with torch.cuda.device(1):
                with torch.no_grad():
                    o_whole = self.model(x_whole)
                    p_ind_whole = o_whole.max(-1)[1]
            
                x_small = F.interpolate(x_whole, (112, 112))
                x_mosaic = torch.zeros_like(x_whole[0])
                for i in range(4):
                    row_num, column_num = i % 2, i // 2
                    x_mosaic[:, column_num*112: (column_num+1)*112, row_num*112: (row_num+1)*112] = x_small[i]
                x_mosaic = x_mosaic.unsqueeze(0)
                    
                with torch.no_grad():
                    o_mosaic = self.model(x_mosaic)
                    
                    ### STOCHASTIC LABEL
                    max_ind = torch.multinomial(o_mosaic[torch.arange(o_mosaic.shape[0]), p_ind_whole].softmax(-1), 1).flatten().detach().cpu().item()
                    
                    p_ind_mosaic = p_ind_whole[max_ind]
            
            (fore_xs, fore_os), fore_diffs, _ = foreground_background_adversarial_attack_for_stepwise_comparison(x_whole[max_ind].unsqueeze(0), self.model, torch.tensor([p_ind_mosaic]), 'foreground', steps=step_num)
            (back_xs, back_os), back_diffs, _ = foreground_background_adversarial_attack_for_stepwise_comparison(x_whole[max_ind].unsqueeze(0), self.model, torch.tensor([p_ind_mosaic]), 'background', steps=step_num)
            
            combined_diffs = combine_insert_remove(back_diffs, fore_diffs)
            combined_sign = torch.where(combined_diffs >= 0., 1., -1.)
            
            for rname in self.relevancy_methods:
                with torch.cuda.device(1):
                    self.relevancy_methods[rname].chosen_index = torch.tensor([p_ind_mosaic])
                    r_mosaic, _, _ = self.relevancy_methods[rname].relevancy(x_mosaic)
                    r_mosaic = norm_function(r_mosaic)

                    self.relevancy_methods[rname].chosen_index = p_ind_whole
                    r_whole, _, _ = self.relevancy_methods[rname].relevancy(x_whole)
                    r_whole = norm_function(r_whole)

                    r_chosen_whole = r_whole[max_ind].unsqueeze(0)
                    single_overlap_score = 0.
                    for index in range(4):
                        if index == max_ind:
                            continue
                        single_overlap_score += 1 - l1_distance(r_chosen_whole, r_whole[index].unsqueeze(0)) / ((r_chosen_whole.flatten(1).sum(-1) + r_whole[index].unsqueeze(0).flatten(1).sum(-1) + 1e-9))
                    single_overlap_score = single_overlap_score / 3
                    
                    r_whole = r_chosen_whole

                    fore_all_start_diffs, back_all_start_diffs = [], []
                    for step_i, (step_fore_x, step_back_x) in enumerate(zip(fore_xs, back_xs)):
                        fx, bx = step_fore_x.to(self.device), step_back_x.to(self.device)

                        self.relevancy_methods[rname].chosen_index = torch.tensor([p_ind_mosaic])
                        with torch.cuda.device(1):
                            back_r, _, _ = self.relevancy_methods[rname].relevancy(bx)
                            fore_r, _, _ = self.relevancy_methods[rname].relevancy(fx)

                        if step_i == 0:
                            r_start = back_r.detach().clone()
                        else:
                            fore_r = normalize_prel(fore_r)
                            fore_start_r = normalize_prel(r_start)
                            back_r = normalize_prel(back_r)
                            back_start_r = normalize_prel(r_start)
                                                
                            fore_start_diff = l1_distance(fore_start_r, fore_r) / ((fore_start_r.flatten(1).sum(-1) + fore_r.flatten(1).sum(-1) + 1e-9))
                            back_start_diff = l1_distance(back_start_r, back_r) / ((back_start_r.flatten(1).sum(-1) + back_r.flatten(1).sum(-1) + 1e-9))

                            fore_all_start_diffs.append(fore_start_diff)
                            back_all_start_diffs.append(back_start_diff)

                    fore_all_start_diffs = torch.stack(fore_all_start_diffs, 1)
                    back_all_start_diffs = torch.stack(back_all_start_diffs, 1)
                    
                    fore_all_start_diffs_os = (fore_os[:, :1] - fore_os[:, 1:]).abs() / (fore_os[:, :1].abs() + fore_os[:, 1:].abs() + 1e-9)
                    back_all_start_diffs_os = (back_os[:, :1] - back_os[:, 1:]).abs() / (back_os[:, :1].abs() + back_os[:, 1:].abs() + 1e-9)

                    diff_rs = fore_all_start_diffs - back_all_start_diffs
                    diff_os = fore_all_start_diffs_os - back_all_start_diffs_os
                    sequence_score = 2 * (1 - (diff_os - diff_rs).abs().sum(-1) / (diff_os.abs().sum(-1) + diff_rs.abs().sum(-1) + 1e-9)) - 1
                    
                    ### CALC SCORE
                    
                    overlap_scores = 0.
                    
                    row_num, column_num = max_ind % 2, max_ind // 2
                    r_single_mosaic = r_mosaic[:, :, column_num*112: (column_num+1)*112, row_num*112: (row_num+1)*112]

                    r_other_mosaic = r_mosaic.detach().clone()
                    
                    r_single_whole = F.interpolate(r_whole[0].unsqueeze(0), (112, 112))
                    r_single_whole = norm_function(r_single_whole)
                    
                    overlap_score = 1 - 2 * ((r_single_mosaic.flatten(1) - r_single_whole.flatten(1)).abs() * r_single_whole.flatten(1)).sum(-1) / (r_single_whole.flatten(1).sum(-1) + 1e-9)
                    
                    chosen_softmax_values = o_whole[max_ind, p_ind_whole].softmax(-1)
                    penalty_mosaic = torch.zeros_like(r_other_mosaic)
                    for curr_index in range(4):
                        curr_column, curr_row = curr_index % 2, curr_index // 2
                        penalty_ratio = chosen_softmax_values[curr_index] / (chosen_softmax_values[max_ind] + 1e-9)
                        penalty_ratio = penalty_ratio.detach().cpu().to(penalty_mosaic.device)

                        if curr_index == max_ind:
                            penalty_mosaic[:, :, curr_row*112: (curr_row+1)*112, curr_column*112: (curr_column+1)*112] = 1.
                        else:
                            penalty_mosaic[:, :, curr_row*112: (curr_row+1)*112, curr_column*112: (curr_column+1)*112] = 2 * penalty_ratio - 1
                    penalty_mosaic[:, :, 111: 113, :] = 1.
                    penalty_mosaic[:, :, :, 111: 113] = 1.

                    overlap_penalty = (r_other_mosaic.flatten(1) * penalty_mosaic.flatten(1)).sum(-1) / (r_mosaic.flatten(1).sum(-1) + 1e-9)
                    combined_overlap_score = (r_whole[0].unsqueeze(0).flatten(1) * combined_sign.flatten(1)).sum(-1) / (r_whole[0].unsqueeze(0).flatten(1).sum(-1) + 1e-9)

                    full_overlap_score = overlap_score + overlap_penalty + sequence_score + combined_overlap_score
                    
                    score_names = ['overlap_score', 'overlap_penalty', 'combined_score', 'sequence_score', 'single_overlap_score']
                    if rname not in detailed_scores:
                        detailed_scores[rname] = {score_name: [] for score_name in score_names}
                    
                    for score, score_name in zip([overlap_score, overlap_penalty, combined_overlap_score, sequence_score, (1 - single_overlap_score)], score_names):
                        detailed_scores[rname][score_name].append(score)

                    overlap_scores += full_overlap_score
                if rname not in final_scores:
                    final_scores[rname] = []
                final_scores[rname].append(overlap_scores)
            
            if example_num + 1 == num_examples:
                break
        
        self.final_scores = final_scores
        
    def plot_results(self):
        for i, (rname, scores) in enumerate(self.final_scores.items()):
            score = torch.cat(scores).mean()
            plt.bar(i, score, label=rname)
        plt.legend()
        plt.show()