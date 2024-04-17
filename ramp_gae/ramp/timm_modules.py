from ramp_gae.ramp.base import *
import timm
import copy


class TimmReLU(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.module = module
    
    def forward(self, x):
        h = self.module(x)
        self.x, self.h, self.ha = x, h, h
        
        return h


class TimmLinear(FFModule):
    def __init__(self, module):
        super().__init__()
        
        out_f, in_f = module.weight.shape
        bias = module.bias is not None
        self.module = FFLinear(in_f, out_f, bias=bias)
        
        self.module.weight = nn.Parameter(module.weight.data)
        if bias:
            self.module.bias = nn.Parameter(module.bias.data)
            
    def forward(self, x):
        x = self.module(x)
        
        return x
    
    def backward_prel(self, prel, rule):
        prel = self.module.backward_prel(prel, rule)
        
        return prel.detach()



class TimmConv2d(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.module = module
        self.abs_module = copy.deepcopy(module)
        self.abs_module.weight = nn.Parameter(self.abs_module.weight.data.abs())
        if module.bias is not None:
            self.abs_module.bias = nn.Parameter(self.abs_module.bias.data.abs())
        
    def forward(self, x):
        h = self.module(x)
        ha = self.abs_module(x.abs())
        
        self.x, self.h, self.ha = x, h, ha
        
        return h

class TimmHead(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.global_pool = TimmSelectAdaptivePool2d(module.global_pool)
        self.fc = TimmLinear(module.fc)
    
    def forward(self, x):
        x = self.global_pool(x)
        x = self.fc(x)
        
        return x
    
    def backward_prel(self, prel, rule):
        prel = self.fc.backward_prel(prel, rule)
        prel = self.global_pool.backward_prel(prel, rule)
        
        return prel

class TimmPreLogits(FFModule):
    def __init__(self, module, drop_rate):
        super().__init__()
        
        self.fc1 = TimmConv2d(module.fc1)
        self.act1 = TimmReLU(module.act1)
        self.fc2 = TimmConv2d(module.fc2)
        self.act2 = TimmReLU(module.act2)
        self.drop_rate = drop_rate
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc2(x)
        x = self.act2(x)
        
        return x
    
    def backward_prel(self, prel, rule):
        prel = self.act2.backward_prel(prel, rule)
        prel = self.fc2.backward_prel(prel, rule)
        prel = self.act1.backward_prel(prel, rule)
        prel = self.fc1.backward_prel(prel, rule)
        
        return prel

class TimmFeatures(FFModule):
    def __init__(self, module):
        super().__init__()
        
        sublayer_dict = {
            'Conv2d': TimmConv2d,
            'ReLU': TimmReLU,
            'MaxPool2d': TimmMaxPool2d,
        }
        
        sublayers = []
        for sublayer in module.children():
            sublayer_name = sublayer.__class__.__name__
            sublayers.append(sublayer_dict[sublayer_name](sublayer))
        self.model = FFSequential(*sublayers)
    
    def forward(self, x):
        return self.model(x)
    
    def backward_prel(self, prel, rule):
        return self.model.backward_prel(prel, rule).detach()
    
class TimmMaxPool2d(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.module = module
        
    def forward(self, x):
        h = self.module(x)
        
        self.x, self.h, self.ha = x, h, h.abs()
        
        return h

    def backward_prel(self, prel, rule):
        x, h, ha = self.x, self.h, self.ha
        
        h_sign = torch.where(h >= 0, 1, -1)
        h_eta = torch.where(h == 0, 1e-9, 0)
        ha_eta = torch.where(ha == 0, 1e-9, 0)
        
        if rule == 'intfill':
            rel = new_intfill_rule(ha, h, x, prel, divide=False)
        if rule == 'intline':
            rel = new_intline_rule(ha, h, x, prel, divide=False)
        if rule == 'intneg':
            rel = torch.autograd.grad(ha + h, x, prel * h_sign, retain_graph=True)[0] * x - torch.autograd.grad(-ha + h, x, prel * h_sign)[0] * x
        if rule == 'intabs':
            rel = torch.autograd.grad(ha + h, x, prel * h_sign)[0] * x
        if rule == 'intmax':
            rel = torch.autograd.grad(ha + h, x, prel * h_sign)[0] * x
        if rule == 'alpha1beta0':
            rel = new_alpha1beta0_rule(ha, h, x, prel)
        
        self.x, self.h, self.ha = None, None, None
        
        return rel.detach()

class TimmSelectAdaptivePool2d(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.module = module
        
    def forward(self, x):
        h = self.module(x)
        ha = self.module(x.abs())
        
        self.x, self.h, self.ha = x, h, ha
        
        return h
    
    def backward_prel(self, prel, rule):
        x, h, ha = self.x, self.h, self.ha
        
        h_sign = torch.where(h >= 0, 1, -1)
        h_eta = torch.where(h == 0, 1e-9, 0)
        ha_eta = torch.where(ha == 0, 1e-9, 0)
        
        if rule == 'intfill':
            rel = new_intfill_rule(ha, h, x, prel, divide=False)
        if rule == 'intline':
            rel = new_intline_rule(ha, h, x, prel, divide=False)
        if rule == 'intneg':
            rel = torch.autograd.grad(ha + h, x, prel * h_sign, retain_graph=True)[0] * x - torch.autograd.grad(-ha + h, x, prel * h_sign)[0] * x
        if rule == 'intabs':
            rel = torch.autograd.grad(ha + h, x, prel * h_sign)[0] * x
        if rule == 'intmax':
            rel = torch.autograd.grad(ha + h, x, prel * h_sign)[0] * x
        if rule == 'alpha1beta0':
            rel = new_alpha1beta0_rule(ha, h, x, prel)
        
        self.x, self.h, self.ha = None, None, None
        
        return rel.detach()
    

class TimmBatchNorm2d(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.module = module
    
    def forward(self, x):
        if self.module.training:
            raise ValueError('LRP is only supported in eval mode')
        
        xmean, xvar = self.module.running_mean[None, :, None, None], self.module.running_var[None, :, None, None]
        w, b = self.module.weight[None, :, None, None], self.module.bias[None, :, None, None]
        xm = x - xmean
        h1 = xm / (xvar + self.module.eps).pow(0.5)
        h1a = (x.abs() - xmean.abs()) / (xvar.abs() + self.module.eps).pow(0.5)
        
        h2 = h1 * w + b
        h2a = h1.abs() * w.abs() + b.abs()
        
        #print('bnorm', 1 - (h1.flatten(1).abs() - h1a.flatten(1)).abs().sum(-1) / (h1.flatten(1).abs().sum(-1) + h1a.flatten(1).sum(-1) + 1e-9))
        #print('bnorm', 1 - (h2.flatten(1).abs() - h2a.flatten(1)).abs().sum(-1) / (h2.flatten(1).abs().sum(-1) + h2a.flatten(1).sum(-1) + 1e-9))
        
        self.x, self.xm, self.h1, self.h1a, self.h2, self.h2a = x, xm, h1, h1a, h2, h2a
        
        return h2
    
    def backward_prel(self, prel, rule):
        x, xm, h1, h1a, h2, h2a = self.x, self.xm, self.h1, self.h1a, self.h2, self.h2a
        
        h2_sign = torch.where(h2 >= 0, 1, -1)
        h2_eta = torch.where(h2 == 0, 1e-9, 0)
        h1_sign = torch.where(h1 >= 0, 1, -1)
        h1_eta = torch.where(h1 == 0, 1e-9, 0)
        
        if rule == 'intfill':
            rel = torch.autograd.grad(h2a, h1, prel * h2_sign / (h2 + h2_eta))[0] * h1
            rel = torch.autograd.grad(h1a, x, rel * h1_sign / (h1 + h1_eta))[0] * x
        if rule == 'intline':
            rel = new_intline_rule(h2a, h2, h1, prel)
            rel = new_intline_rule(h1a, h1, x, rel)
        
        self.x, self.xm, self.h1, self.h1a, self.h2, self.h2a = [None] * 6
            
        return rel.detach()
    

class TimmBasicBlock(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.conv1 = TimmConv2d(module.conv1)
        self.bn1 = TimmBatchNorm2d(module.bn1)
        self.act1 = TimmReLU(module.act1)
        self.conv2 = TimmConv2d(module.conv2)
        self.bn2 = TimmBatchNorm2d(module.bn2)
        self.act2 = TimmReLU(module.act2)
        
        self.split_residual = FFSplitResidual()
        self.combine_residual = FFCombineResidual()
        
        if module.downsample is not None:
            self.downsample = True
            self.downsample_conv = TimmConv2d(module.downsample[0])
            self.downsample_bn = TimmBatchNorm2d(module.downsample[1])
        else:
            self.downsample = False
        
    def forward(self, x):
        shortcut, x = self.split_residual(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample:
            shortcut = self.downsample_conv(shortcut)
            shortcut = self.downsample_bn(shortcut)
        
        x = self.combine_residual(shortcut, x)
        
        x = self.act2(x)
        
        return x
    
    def backward_prel(self, prel, rule):
        prel = self.act2.backward_prel(prel, rule)
        #prel = normalize_abs_sum_to_one(prel).detach()
        prel_shortcut, prel_x = self.combine_residual.backward_prel(prel, rule)
        ss = prel_shortcut.flatten(1).abs().sum(-1)
        xs = prel_x.flatten(1).abs().sum(-1)
        if rule == 'intline':
            ssp, ssn = prel_shortcut.flatten(1).clamp(min=0).sum(-1), prel_shortcut.flatten(1).clamp(max=0).sum(-1)
            xsp, xsn = prel_x.flatten(1).clamp(min=0).sum(-1), prel_x.flatten(1).clamp(max=0).sum(-1)
        #print(ss, xs)
        #prel_shortcut, prel_x = normalize_abs_sum_to_one(prel_shortcut).detach(), normalize_abs_sum_to_one(prel_x).detach()
        
        if self.downsample:
            prel_shortcut = self.downsample_bn.backward_prel(prel_shortcut, rule)
            #prel_shortcut = normalize_abs_sum_to_one(prel_shortcut).detach()
            prel_shortcut = self.downsample_conv.backward_prel(prel_shortcut, rule)
            #prel_shortcut = normalize_abs_sum_to_one(prel_shortcut).detach()
        
        prel_x = self.bn2.backward_prel(prel_x, rule)
        #prel_x = normalize_abs_sum_to_one(prel_x).detach()
        prel_x = self.conv2.backward_prel(prel_x, rule)
        #prel_x = normalize_abs_sum_to_one(prel_x).detach()
        prel_x = self.act1.backward_prel(prel_x, rule)
        #prel_x = normalize_abs_sum_to_one(prel_x).detach()
        prel_x = self.bn1.backward_prel(prel_x, rule)
        #prel_x = normalize_abs_sum_to_one(prel_x).detach()
        prel_x = self.conv1.backward_prel(prel_x, rule)
        #prel_x = normalize_abs_sum_to_one(prel_x).detach()

        if rule == 'intline':
            prel = prel_shortcut + prel_x
            #prel = self.split_residual.backward_prel(prel_shortcut * ss[:, None, None, None], prel_x * xs[:, None, None, None], rule)
        else:
            prel_shortcut, prel_x = normalize_abs_sum_to_one(prel_shortcut).detach(), normalize_abs_sum_to_one(prel_x).detach()
            prel = self.split_residual.backward_prel(prel_shortcut * ss[:, None, None, None], prel_x * xs[:, None, None, None], rule)

        return prel

class TimmBottleneck(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.conv1 = TimmConv2d(module.conv1)
        self.bn1 = TimmBatchNorm2d(module.bn1)
        self.act1 = TimmReLU(module.act1)
        self.conv2 = TimmConv2d(module.conv2)
        self.bn2 = TimmBatchNorm2d(module.bn2)
        self.act2 = TimmReLU(module.act2)
        self.conv3 = TimmConv2d(module.conv3)
        self.bn3 = TimmBatchNorm2d(module.bn3)
        self.act3 = TimmReLU(module.act3)
        
        self.split_residual = FFSplitResidual()
        self.combine_residual = FFCombineResidual()
        
        if module.downsample is not None:
            self.downsample = True
            self.downsample_conv = TimmConv2d(module.downsample[0])
            self.downsample_bn = TimmBatchNorm2d(module.downsample[1])
        else:
            self.downsample = False
        
    def forward(self, x):
        shortcut, x = self.split_residual(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.downsample:
            shortcut = self.downsample_conv(shortcut)
            shortcut = self.downsample_bn(shortcut)
        
        x = self.combine_residual(shortcut, x)
        
        x = self.act3(x)
        
        return x
    
    def backward_prel(self, prel, rule):
        prel = self.act3.backward_prel(prel, rule)
        #print('act', prel.sum())
        #prel = normalize_abs_sum_to_one(prel).detach()
        prel_shortcut, prel_x = self.combine_residual.backward_prel(prel, rule)
        
        #print('combine_res', (prel_shortcut + prel_x).sum())
        
        ss = prel_shortcut.flatten(1).abs().sum(-1)
        xs = prel_x.flatten(1).abs().sum(-1)
        
        if rule == 'intline':
            ssp, ssn = prel_shortcut.flatten(1).clamp(min=0).sum(-1), prel_shortcut.flatten(1).clamp(max=0).sum(-1)
            xsp, xsn = prel_x.flatten(1).clamp(min=0).sum(-1), prel_x.flatten(1).clamp(max=0).sum(-1)
        #print(ss, xs)
        #prel_shortcut, prel_x = normalize_abs_sum_to_one(prel_shortcut).detach(), normalize_abs_sum_to_one(prel_x).detach()
        
        if self.downsample:
            #print('short_pre_bn', prel_shortcut.sum())
            prel_shortcut = self.downsample_bn.backward_prel(prel_shortcut, rule)
            #print('short_post_bn', prel_shortcut.sum())
            #prel_shortcut = normalize_abs_sum_to_one(prel_shortcut).detach()
            prel_shortcut = self.downsample_conv.backward_prel(prel_shortcut, rule)
            #print('short_post_downsample_conv', prel_shortcut.sum())
            #prel_shortcut = normalize_abs_sum_to_one(prel_shortcut).detach()
        
        #print('x_pre_shit', prel_x.sum())
        prel_x = self.bn3.backward_prel(prel_x, rule)
        #prel_x = normalize_abs_sum_to_one(prel_x).detach()
        prel_x = self.conv3.backward_prel(prel_x, rule)
        #prel_x = normalize_abs_sum_to_one(prel_x).detach()
        prel_x = self.act2.backward_prel(prel_x, rule)
        #prel_x = normalize_abs_sum_to_one(prel_x).detach()
        prel_x = self.bn2.backward_prel(prel_x, rule)
        #prel_x = normalize_abs_sum_to_one(prel_x).detach()
        prel_x = self.conv2.backward_prel(prel_x, rule)
        #prel_x = normalize_abs_sum_to_one(prel_x).detach()
        prel_x = self.act1.backward_prel(prel_x, rule)
        #prel_x = normalize_abs_sum_to_one(prel_x).detach()
        prel_x = self.bn1.backward_prel(prel_x, rule)
        #prel_x = normalize_abs_sum_to_one(prel_x).detach()
        prel_x = self.conv1.backward_prel(prel_x, rule)
        #prel_x = normalize_abs_sum_to_one(prel_x).detach()
        #print('x_post_shit', prel_x.sum())
        #print('short_pre_sum', prel_shortcut.sum())
        
        if rule == 'intline':
            prel = prel_shortcut + prel_x
            #prel_shortcut, prel_x = normalize_abs_sum_to_one(prel_shortcut).detach(), normalize_abs_sum_to_one(prel_x).detach()
            #prel = self.split_residual.backward_prel(prel_shortcut * ss[:, None, None, None], prel_x * xs[:, None, None, None], rule)
        else:
            prel_shortcut, prel_x = normalize_abs_sum_to_one(prel_shortcut).detach(), normalize_abs_sum_to_one(prel_x).detach()
            prel = self.split_residual.backward_prel(prel_shortcut * ss[:, None, None, None], prel_x * xs[:, None, None, None], rule)
        
        #print('prel_after_sum', prel.sum())
        return prel

class TimmResNetLayer(FFModule):
    def __init__(self, module):
        super().__init__()
        
        sublayers = []
        for sublayer in module.children():
            if isinstance(sublayer, timm.models.resnet.BasicBlock):
                sublayers.append(TimmBasicBlock(sublayer))
            elif isinstance(sublayer, timm.models.resnet.Bottleneck):
                sublayers.append(TimmBottleneck(sublayer))
        self.model = FFSequential(*sublayers)
    
    def forward(self, x):
        return self.model(x)
    
    def backward_prel(self, prel, rule):
        return self.model.backward_prel(prel, rule).detach()
    
class TimmMaxPool2d(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.module = module
        
    def forward(self, x):
        h = self.module(x)
        
        self.x, self.h, self.ha = x, h, h.abs()
        
        return h
    
    def backward_prel(self, prel, rule):
        x, h, ha = self.x, self.h, self.ha
        
        h_sign = torch.where(h >= 0, 1, -1)
        h_eta = torch.where(h == 0, 1e-9, 0)
        ha_eta = torch.where(ha == 0, 1e-9, 0)
        
        if rule == 'intfill':
            rel = torch.autograd.grad(ha, x, prel * h_sign)[0] * x
            #rel = torch.autograd.grad(ha, x, prel * h_sign / (h + h_eta))[0] * x
        if rule == 'intline':
            rel = new_intline_rule(ha, h, x, prel, divide=True)
        
        self.x, self.h, self.ha = None, None, None
        
        return rel.detach()

class TimmSelectAdaptivePool2d(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.module = module
        
    def forward(self, x):
        h = self.module(x)
        ha = self.module(x.abs())
        
        self.x, self.h, self.ha = x, h, ha
        
        return h
    
    def backward_prel(self, prel, rule):
        x, h, ha = self.x, self.h, self.ha
        
        h_sign = torch.where(h >= 0, 1, -1)
        h_eta = torch.where(h == 0, 1e-9, 0)
        ha_eta = torch.where(ha == 0, 1e-9, 0)
        
        if rule == 'intfill':
            rel = torch.autograd.grad(ha, x, prel * h_sign)[0] * x
            #rel = torch.autograd.grad(ha, x, prel * h_sign / (h + h_eta))[0] * x
        if rule == 'intline':
            rel = new_intline_rule(ha, h, x, prel, divide=True)
        
        self.x, self.h, self.ha = None, None, None
        
        return rel.detach()
    

class TimmLayerNorm(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.module = module
        
    def forward(self, x):
        x_mean = x.mean(-1, keepdim=True).detach()
        x_var = x.var(-1, keepdim=True, unbiased=False).detach()
        h1 = (x - x_mean) / (x_var + self.module.eps).pow(0.5)
        h1a = (x.abs() + x_mean.abs()) / (x_var.abs() + self.module.eps).pow(0.5)
        
        h2 = h1 * self.module.weight + self.module.bias
        h2a = h1.abs() * self.module.weight.abs() + self.module.bias.abs()
        
        #h = self.module(x)
        
        self.x, self.h1, self.h1a, self.h2, self.h2a = x, h1, h1a, h2, h2a
        
        return h2
    
    def backward_prel(self, prel, rule):
        x, h1, h1a, h2, h2a = self.x, self.h1, self.h1a, self.h2, self.h2a
        
        h2_sign = get_sign(h2)#torch.where(h2 >= 0, 1, -1)
        h2_eta = torch.where(h2 == 0, 1e-9, 0)
        h1_sign = get_sign(h1)#torch.where(h1 >= 0, 1, -1)
        h1_eta = torch.where(h1 == 0, 1e-9, 0)
        
        if rule == 'intfill':
            rel = torch.autograd.grad(h2a, h1, prel * h2_sign / (h2 + h2_eta))[0] * h1
            rel = torch.autograd.grad(h1a, x, rel * h1_sign / (h1 + h1_eta))[0] * x
        elif rule == 'intline':
            rel = torch.autograd.grad(h2a + h2, h1, prel * h2_sign / (h2 + h2_eta))[0] * h1
            rel = torch.autograd.grad(h1a + h1, x, rel * h1_sign / (h1 + h1_eta))[0] * x
            #rel = torch.autograd.grad(h2a + h2, h1, prel * h2_sign)[0] * h1
            #rel = torch.autograd.grad(h1a + h1, x, rel * h1_sign)[0] * x
        
        self.x, self.h1, self.h1a, self.h2, self.h2a = [None] * 5
            
        return rel.detach()
    
    
class TimmGELU(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.module = module
        
    def forward(self, x):
        h = self.module(x)
        self.x, self.h, self.ha = x, h, h.abs()
        
        return h

class TimmLinear(FFModule):
    def __init__(self, module):
        super().__init__()
        
        out_f, in_f = module.weight.shape
        bias = module.bias is not None
        self.module = FFLinear(in_f, out_f, bias=bias)
        
        self.module.weight = nn.Parameter(module.weight.data)
        if bias:
            self.module.bias = nn.Parameter(module.bias.data)
            
    def forward(self, x):
        x = self.module(x)
        
        return x
    
    def backward_prel(self, prel, rule):
        prel = self.module.backward_prel(prel, rule)
        
        return prel

class TimmConv2d(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.module = module
        self.abs_module = copy.deepcopy(module)
        self.abs_module.weight = nn.Parameter(self.abs_module.weight.data.abs())
        if module.bias is not None:
            self.abs_module.bias = nn.Parameter(self.abs_module.bias.data.abs())
        
    def forward(self, x):
        h = self.module(x)
        ha = self.abs_module(x.abs())
        
        self.x, self.h, self.ha = x, h, ha
        
        return h


class TimmLayerNorm(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.module = module
        
    def forward(self, x):
        x_mean = x.mean(-1, keepdim=True).detach()
        x_var = x.var(-1, keepdim=True, unbiased=False).detach()
        
        xm = x - x_mean
        xma = x.abs() + x_mean.abs()
        
        h1 = xm / (x_var + self.module.eps).pow(0.5)
        h1a = xm.abs() / (x_var.abs() + self.module.eps).pow(0.5)
        
        h2 = h1 * self.module.weight + self.module.bias
        h2a = h1.abs() * self.module.weight.abs() + self.module.bias.abs()
        
        #print('lnorm', 1 - (h1.flatten(1).abs() - h1a.flatten(1)).abs().sum(-1) / (h1.flatten(1).abs().sum(-1) + h1a.flatten(1).sum(-1) + 1e-9))
        #print('lnorm', 1 - (h2.flatten(1).abs() - h2a.flatten(1)).abs().sum(-1) / (h2.flatten(1).abs().sum(-1) + h2a.flatten(1).sum(-1) + 1e-9))
        
        self.x, self.x_mean, self.xm, self.xma, self.h1, self.h1a, self.h2, self.h2a = x, x_mean, xm, xma, h1, h1a, h2, h2a
        
        return h2
    
    def backward_prel(self, prel, rule):
        x, x_mean, xm, xma, h1, h1a, h2, h2a = self.x, self.x_mean, self.xm, self.xma, self.h1, self.h1a, self.h2, self.h2a
        
        h2_sign = get_sign(h2)#torch.where(h2 >= 0, 1, -1)
        h2_eta = torch.where(h2 == 0, 1e-9, 0)
        h1_sign = get_sign(h1)#torch.where(h1 >= 0, 1, -1)
        h1_eta = torch.where(h1 == 0, 1e-9, 0)
        xm_sign = get_sign(xm)#torch.where(h1 >= 0, 1, -1)
        xm_eta = torch.where(xm == 0, 1e-9, 0)
        
        if rule == 'intfill':
            rel = torch.autograd.grad(h2a + h2, x, prel * h2_sign / (h2 + h2_eta))[0] * x
            rel = rel.relu()
        elif 'intline' in rule:
            rel = new_intline_rule(h2a, h2, x, prel)
        elif rule == 'alpha1beta0':
            #print(prel.sum())
            """rel = torch.autograd.grad(h2a + h2, x, prel * h2_sign / (h2 + h2_eta))[0] * x
            rel = rel.relu()"""
            rel = new_alpha1beta0_rule(h2a, h2, x, prel)
            rel = rel.relu()
        
        self.x, self.x_mean, self.xm, self.xma, self.h1, self.h1a, self.h2, self.h2a = [None] * 8
            
        return rel.detach()

class TimmSoftmax(FFModule):
    def __init__(self, dim):
        super().__init__()
        
        self.dim = dim
        
    def forward(self, x):
        x_max = x.max(-1, keepdim=True)[0].detach()
        
        xm = x - x_max
        xma = x.abs() + x_max.abs()
        xme = xm.exp()
        h = xme / xme.sum(self.dim, keepdim=True).detach()
        
        self.x, self.xm, self.xma, self.xme, self.h = x, xm, xma, xme, h
        
        return h
        
    def backward_prel(self, prel, rule):
        x, xm, xma, xme, h = self.x, self.xm, self.xma, self.xme, self.h
        
        h_sign = get_sign(h)#torch.where(h >= 0, 1, -1)
        h_eta = torch.where(h == 0, 1e-9, 0)
        xm_sign = get_sign(xm)#torch.where(h >= 0, 1, -1)
        xm_eta = torch.where(xm == 0, 1e-9, 0)
        xme_sign = get_sign(xme)#torch.where(h >= 0, 1, -1)
        xme_eta = torch.where(xme == 0, 1e-9, 0)
        
        if rule == 'intfill':
            rel = torch.autograd.grad(h, x, prel * h_sign / (h + h_eta))[0] * x
            rel = rel.relu()
        elif 'intline' in rule:          
            rel = new_intline_rule(h, h, x, prel)
        elif rule == 'alpha1beta0':
            #print(prel.sum())
            """rel = torch.autograd.grad(h2a + h2, x, prel * h2_sign / (h2 + h2_eta))[0] * x
            rel = rel.relu()"""
            rel = new_alpha1beta0_rule(h, h, x, prel)
            rel = rel.relu()
            
        self.x, self.xm, self.xma, self.xme, self.h = [None] * 5
            
        return rel.detach()


class TimmAttentionMultiply(FFModule):
    def __init__(self, scale):
        super().__init__()
        
        self.scale = scale
        
    def forward(self, q, k):
        h = (q @ k.transpose(-2, -1)) * self.scale
        ha = (q.abs() @ k.abs().transpose(-2, -1)) * self.scale
        
        self.q, self.k, self.h, self.ha = q, k, h, ha
        
        return h
        
    def backward_prel(self, prel, rule):
        q, k, h, ha = self.q, self.k, self.h, self.ha
        
        h_sign = get_sign(h)#torch.where(h >= 0, 1, -1)
        h_eta = torch.where(h == 0, 1e-9, 0)
        
        if rule == 'intfill':
            #rel_q = torch.autograd.grad(ha, q, prel * h_sign / (h + h_eta), retain_graph=True)[0] * q
            #rel_k = torch.autograd.grad(ha, k, prel * h_sign / (h + h_eta))[0] * k
            #rel_q = torch.autograd.grad(ha, q, prel, retain_graph=True)[0] * q
            #rel_k = torch.autograd.grad(ha, k, prel)[0] * k
            #rel_q = torch.autograd.grad(ha, q, prel * h_sign, retain_graph=True)[0] * q
            #rel_k = torch.autograd.grad(ha, k, prel * h_sign)[0] * k
            rel_q = torch.autograd.grad(ha + h, q, prel * h_sign / (h + h_eta), retain_graph=True)[0] * q
            rel_k = torch.autograd.grad(ha + h, k, prel * h_sign / (h + h_eta))[0] * k
        elif 'intline' in rule:
            #rel_q = new_intline_rule(ha, h, q, prel, retain_graph=True)
            #rel_k = new_intline_rule(ha, h, k, prel)
            rel_q, rel_k = double_new_intline_rule(ha, h, q, k, prel)
            #rel_q, rel_k = double_new_intline_rule(h, h, q, k, prel)
        elif rule == 'alpha1beta0':
            rel_q = torch.autograd.grad(ha + h, q, prel / (ha + h + h_eta), retain_graph=True)[0] * q
            rel_k = torch.autograd.grad(ha + h, k, prel / (ha + h + h_eta))[0] * k
        
        self.q, self.k, self.h, self.ha = None, None, None, None
        
        #print('attn', prel.min(), rel_q.min(), rel_k.min())
            
        return rel_q.detach(), rel_k.detach()

class TimmAttention(FFModule):
    def __init__(self, module):
        super().__init__()
        
        f = module.qkv.weight.shape[-1]
        heads = module.num_heads
        self.attn_drop = module.attn_drop
        self.proj_drop = module.proj_drop
        self.scale = module.scale
        
        self.num_heads = module.num_heads
        qkv_bias = module.qkv.bias is not None
        self.toq = FFLinear(f, f, bias=qkv_bias)
        self.tok = FFLinear(f, f, bias=qkv_bias)
        self.tov = FFLinear(f, f, bias=qkv_bias)
        
        self.qkv = module.qkv
        self.toq.weight = nn.Parameter(module.qkv.weight.data[:f])
        self.tok.weight = nn.Parameter(module.qkv.weight.data[f:2*f])
        self.tov.weight = nn.Parameter(module.qkv.weight.data[2*f:])
        
        if qkv_bias:
            self.toq.bias = nn.Parameter(module.qkv.bias.data[:f])
            self.tok.bias = nn.Parameter(module.qkv.bias.data[f:2*f])
            self.tov.bias = nn.Parameter(module.qkv.bias.data[2*f:])
        
        self.to_out = TimmLinear(module.proj)
        
        self.attend = TimmSoftmax(-1)
        self.attn_v = FFDynLinear()
        self.attn_mul = TimmAttentionMultiply(self.scale)
        
        self.eheads_q = FFExtractHeads(heads)
        self.eheads_k = FFExtractHeads(heads)
        self.eheads_v = FFExtractHeads(heads)
        self.sheads_v = FFSqueezeHeads(heads)
        
    def forward(self, x):
        q, k, v = self.toq(x), self.tok(x), self.tov(x)
        q, k, v = self.eheads_q(q), self.eheads_k(k), self.eheads_v(v)  # b n (h d) -> b h n d

        attn = self.attn_mul(q, k)
        attn = self.attend(attn)
        attn = self.attn_drop(attn)

        h = self.attn_v(v, attn)
        h = self.sheads_v(h)
        h = self.to_out(h)
        h = self.proj_drop(h)
        
        return h
    
    def backward_prel(self, prel, rule):
        prel = self.to_out.backward_prel(prel.detach(), rule)
        #prel = normalize_abs_sum_to_one(prel).detach()
        prel = self.sheads_v.backward_prel(prel, rule)
        #prel = normalize_abs_sum_to_one(prel).detach()
        prel, prel_w = self.attn_v.backward_prel(prel, rule)
        ps, ws = prel.flatten(1).abs().sum(-1), prel_w.flatten(1).abs().sum(-1)
        
        #prel, prel_w = normalize_abs_sum_to_one(prel).detach(), normalize_abs_sum_to_one(prel_w).detach()
        prel_w = self.attend.backward_prel(prel_w, rule)
        #prel_w = normalize_abs_sum_to_one(prel_w).detach()
        prel_q, prel_k = self.attn_mul.backward_prel(prel_w, rule)
        qs, ks = prel_q.flatten(1).abs().sum(-1), prel_k.flatten(1).abs().sum(-1)
        
        #prel_q, prel_k = normalize_abs_sum_to_one(prel_q).detach(), normalize_abs_sum_to_one(prel_k).detach()
        prel_q, prel_k = self.eheads_q.backward_prel(prel_q, rule), self.eheads_k.backward_prel(prel_k, rule)
        #prel_q, prel_k = normalize_abs_sum_to_one(prel_q).detach(), normalize_abs_sum_to_one(prel_k).detach()
        prel_q, prel_k = self.toq.backward_prel(prel_q, rule), self.tok.backward_prel(prel_k, rule)
        #prel_q, prel_k = normalize_abs_sum_to_one(prel_q).detach(), normalize_abs_sum_to_one(prel_k).detach()
        prel = self.eheads_v.backward_prel(prel, rule)
        #prel = normalize_abs_sum_to_one(prel).detach()
        prel = self.tov.backward_prel(prel, rule)
        #prel = normalize_abs_sum_to_one(prel).detach()
        
        #prel = prel_k
        #prel = prel + prel_q + prel_k
        #prel = normalize_abs_sum_to_one(prel) + normalize_abs_sum_to_one(prel_q) + normalize_abs_sum_to_one(prel_k)
        #print(prel.flatten().sum(), prel_q.flatten().sum(), prel_k.flatten().sum())
        if 'value_only' in rule:
            prel = prel
        elif 'query_key_only' in rule:
            prel = prel_q + prel_k
        elif 'intline' in rule:
            prel = prel + prel_q + prel_k
        else:
            prel_qk = qs[:, None, None] * normalize_abs_sum_to_one(prel_q).detach() + ks[:, None, None] * normalize_abs_sum_to_one(prel_k).detach()
            prel = ps[:, None, None] * normalize_abs_sum_to_one(prel).detach() + ws[:, None, None] * normalize_abs_sum_to_one(prel_qk).detach()
        
        #prel = normalize_abs_sum_to_one(prel).detach()
        
        return prel


class TimmMlp(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.fc1 = TimmLinear(module.fc1)
        self.fc2 = TimmLinear(module.fc2)
            
        self.act = TimmGELU(module.act)
        self.drop = module.drop
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x
    
    def backward_prel(self, prel, rule):
        prel = self.fc2.backward_prel(prel, rule).detach()
        #prel = normalize_prel(prel).detach()
        prel = self.act.backward_prel(prel, rule).detach()
        #prel = normalize_prel(prel).detach()
        prel = self.fc1.backward_prel(prel, rule).detach()
        #prel = normalize_prel(prel).detach()
        
        return prel

class TimmBlock(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.norm1 = TimmLayerNorm(module.norm1)
        self.norm2 = TimmLayerNorm(module.norm2)
        self.drop_path = module.drop_path
        
        self.attn = TimmAttention(module.attn)
        self.mlp = TimmMlp(module.mlp)
        
        self.res1_split = FFSplitResidual()
        self.res1_combine = FFCombineResidual()
        self.res2_split = FFSplitResidual()
        self.res2_combine = FFCombineResidual()
        
    def forward(self, x):
        x1, x2 = self.res1_split(x)
        x2 = self.drop_path(self.attn(self.norm1(x2)))
        x = self.res1_combine(x1, x2)
        x1, x2 = self.res2_split(x)
        x2 = self.drop_path(self.mlp(self.norm2(x2)))
        x = self.res2_combine(x1, x2)
        
        return x
        
    def backward_prel(self, prel, rule):
        #print('block_in', prel.sum())
        prel_x1, prel_x2 = self.res2_combine.backward_prel(prel.detach(), rule)
        px1 = prel_x1.flatten(1).abs().sum(-1)
        px2 = prel_x2.flatten(1).abs().sum(-1)
        
        #print('res_combine', (prel_x1 + prel_x2).sum())
        #prel_x1, prel_x2 = normalize_abs_sum_to_one(prel_x1).detach(), normalize_abs_sum_to_one(prel_x2).detach()
        prel_x2 = self.mlp.backward_prel(prel_x2, rule)
        #prel_x2 = normalize_abs_sum_to_one(prel_x2).detach()
        prel_x2 = self.norm2.backward_prel(prel_x2, rule)
        #prel_x2 = normalize_abs_sum_to_one(prel_x2).detach()
        
        #print('pre_res1', (prel_x1 + prel_x2).sum())
        
        if 'intline' in rule:
            prel = prel_x1 + prel_x2
        else:
            prel_x1, prel_x2 = normalize_abs_sum_to_one(prel_x1).detach(), normalize_abs_sum_to_one(prel_x2).detach()
            prel = self.res2_split.backward_prel(prel_x1 * px1[:, None, None], prel_x2 * px2[:, None, None], rule)
        #prel = prel_x1 + prel_x2
        
        #print('post_res1', prel.sum())
        
        prel_x1, prel_x2 = self.res1_combine.backward_prel(prel, rule)
        px1 = prel_x1.flatten(1).abs().sum(-1)
        px2 = prel_x2.flatten(1).abs().sum(-1)
        
        #print('res_combine', (prel_x1 + prel_x2).sum())
        
        #prel_x1, prel_x2 = normalize_abs_sum_to_one(prel_x1).detach(), normalize_abs_sum_to_one(prel_x2).detach()
        prel_x2 = self.attn.backward_prel(prel_x2, rule)
        #prel_x2 = normalize_abs_sum_to_one(prel_x2).detach()
        prel_x2 = self.norm1.backward_prel(prel_x2, rule)
        #prel_x2 = normalize_abs_sum_to_one(prel_x2).detach()
        
        #print('pre_res2', (prel_x1 + prel_x2).sum())
        
        if 'intline' in rule:
            prel = prel_x1 + prel_x2
        else:
            prel_x1, prel_x2 = normalize_abs_sum_to_one(prel_x1).detach(), normalize_abs_sum_to_one(prel_x2).detach()
            prel = self.res1_split.backward_prel(prel_x1 * px1[:, None, None], prel_x2 * px2[:, None, None], rule)
        
        #print('post_res2', prel.sum())
        #print('block_out', prel.sum())
        
        return prel


class TimmBlocks(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.blocks = FFSequential(*[TimmBlock(block) for block in list(module.children())])
        
    def forward(self, x):
        x = self.blocks(x)
        
        return x
    
    def backward_prel(self, prel, rule):
        prel = self.blocks.backward_prel(prel, rule)
        
        return prel.detach()
    
    
class TimmFlatten(FFModule):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        self.x_shape = x.shape
        h = x.flatten(2).transpose(1, 2)
        ha = x.flatten(2).transpose(1, 2).abs()
        
        self.x, self.h, self.ha = x, h, ha
        
        return h
    
        """def backward_prel(self, prel, rule):
        rel = prel.transpose(1, 2)
        rel = rel.view(self.x_shape)
        
        self.x, self.h, self.ha = [None] * 3
        
        return rel"""


class TimmPatchEmbed(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.proj = TimmConv2d(module.proj)
        self.norm = module.norm
        self.flatten = TimmFlatten()
        
    def forward(self, x):
        x = self.proj(x)
        x = self.flatten(x)
        x = self.norm(x)
        
        return x
    
    def backward_prel(self, prel, rule):
        prel = self.flatten.backward_prel(prel.detach(), rule)
        #prel = normalize_prel(prel).detach()
        prel = self.proj.backward_prel(prel, rule)
        #prel = normalize_prel(prel).detach()
        
        return prel
    

def subtract_weight(w, p_ind):
    anchor_w = w[p_ind][:, None, :]
    contrastive_w = w.unsqueeze(0).repeat(p_ind.shape[0], 1, 1)
    contrastive_w = contrastive_w - anchor_w
    
    return contrastive_w


def subtract_mean(w, p_ind):
    anchor_w = w.mean(0, keepdim=True)
    contrastive_w = w.unsqueeze(0).repeat(p_ind.shape[0], 1, 1)
    contrastive_w = contrastive_w - anchor_w
    
    return contrastive_w


class TimmClassificationLayer(FFModule):
    def __init__(self, module):
        super().__init__()
        
        out_f, in_f = module.weight.shape
        
        self.weight = nn.Parameter(module.weight.data)
        
        self.bias = None
        if module.bias is not None:
            self.bias = nn.Parameter(module.bias.data)
            
    def forward(self, x, p_ind=None):
        w, b = self.weight, self.bias
        
        if p_ind is None:
            h = F.linear(x, w, b)
            ha = F.linear(x.abs(), w.abs(), b.abs())
        else:
            w_contrastive = subtract_weight(w, p_ind)
            h = torch.bmm(x[:, None, :], w_contrastive.transpose(1, 2)).squeeze(1)
            ha = torch.bmm(x[:, None, :].abs(), w_contrastive.transpose(1, 2).abs()).squeeze(1)
        
        self.x, self.h, self.ha = x, h, ha
        
        return h