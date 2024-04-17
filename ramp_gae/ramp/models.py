from ramp_gae.ramp.base import *
from ramp_gae.ramp.timm_modules import *
from ramp_gae.ramp.huggingface_modules import *


class TimmVGG16(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.features = TimmFeatures(module.features)
        self.pre_logits = TimmPreLogits(module.pre_logits, module.drop_rate)
        self.head = TimmHead(module.head)
        
    def forward(self, x):
        x = self.features(x)
        x = self.pre_logits(x)
        x = self.head(x)
        
        return x
    
    def backward_prel(self, prel, rule):
        prel = self.head.backward_prel(prel, rule)
        prel = self.pre_logits.backward_prel(prel, rule)
        prel = self.features.backward_prel(prel, rule)
        
        return prel


class TimmResNet(FFModule):
    def __init__(self, module):
        super().__init__()
        
        self.conv1 = TimmConv2d(module.conv1)
        self.bn1 = TimmBatchNorm2d(module.bn1)
        self.act1 = TimmReLU(module.act1)
        self.maxpool = TimmMaxPool2d(module.maxpool)
        layers = [TimmResNetLayer(module.__getattr__(layer_name)) for layer_name in list(filter(lambda k: 'layer' in k, dir(module)))]
        self.layers = FFSequential(*layers)
        self.global_pool = TimmSelectAdaptivePool2d(module.global_pool)
        self.drop_rate = module.drop_rate
        self.training = module.training
        self.fc = TimmLinear(module.fc)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.layers(x)
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        
        return x
    
    def backward_prel(self, prel, rule):
        #print('orig', prel.sum())
        prel = self.fc.backward_prel(prel, rule)
        #prel = normalize_abs_sum_to_one(prel).detach()
        #print('fc', prel.sum())
        prel = self.global_pool.backward_prel(prel, rule)
        #prel = normalize_abs_sum_to_one(prel).detach()
        #print('avgpool', prel.sum())
        prel = self.layers.backward_prel(prel, rule)
        #prel = normalize_abs_sum_to_one(prel).detach()
        #print('layers', prel.sum())
        prel = self.maxpool.backward_prel(prel, rule)
        #prel = normalize_abs_sum_to_one(prel).detach()
        #print('maxpool', prel.sum())
        prel = self.act1.backward_prel(prel, rule)
        #prel = normalize_abs_sum_to_one(prel).detach()
        #print('act', prel.sum())
        prel = self.bn1.backward_prel(prel, rule)
        #prel = normalize_abs_sum_to_one(prel).detach()
        #print('bn', prel.sum())
        prel = self.conv1.backward_prel(prel, rule)
        #prel = normalize_abs_sum_to_one(prel).detach()
        #print('conv1', prel.sum())
        
        return prel


class TimmVisionTransformer(FFModule):
    def __init__(self, model):
        super().__init__()
        
        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.pos_drop = model.pos_drop
        self.norm = TimmLayerNorm(model.norm)
        
        self.patch_embed = TimmPatchEmbed(model.patch_embed)
        self.blocks = TimmBlocks(model.blocks)
        self.add_pos_emb = FFCombineResidual()#FFResidualConnection()
        self.cls_concat = FFConcat(self.cls_token)
        self.choose_token = FFChooseToken()
        
        self.head = TimmLinear(model.head)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.cls_concat(x)
        x = self.add_pos_emb(x, self.pos_embed)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.choose_token(x)
        x = self.head(x)
        
        return x
    
    def backward_prel(self, prel, rule):
        prel = self.head.backward_prel(prel.detach(), rule)
        #prel = normalize_prel(prel).detach()
        prel = self.choose_token.backward_prel(prel, rule)
        #prel = normalize_prel(prel).detach()
        prel = self.norm.backward_prel(prel, rule)
        #prel = normalize_prel(prel).detach()
        prel = self.blocks.backward_prel(prel, rule)
        #prel = normalize_prel(prel).detach()
        #prel, prel_pos, prel_ratio, pos_ratio = self.add_pos_emb.backward_prel(prel, rule)
        prel, prel_pos = self.add_pos_emb.backward_prel(prel, rule)
        #prel = normalize_prel(prel).detach()
        #prel = prel * prel_ratio
        #prel = normalize_prel(prel).detach()
        prel = self.cls_concat.backward_prel(prel, rule)
        prel_prepatch = prel.detach().clone()
        #prel_prepatch = normalize_prel(prel).detach()
        prel = self.patch_embed.backward_prel(prel_prepatch, rule)
        #prel = normalize_prel(prel).detach()
        
        return prel#, prel_prepatch


class RAMPDistilBertForSequenceClassification(FFModule):
    def __init__(self, model):
        super().__init__()
        
        self.distilbert = HFDistilBert(model.distilbert)
        self.choose_token = FFChooseToken()
        self.pre_classifier = TimmLinear(model.pre_classifier)
        self.act = TimmReLU(nn.ReLU())
        self.dropout = model.dropout
        self.classifier = TimmLinear(model.classifier)
    
    def forward(self, input_ids, attention_mask):
        h = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        h = self.choose_token(h)
        h = self.pre_classifier(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.classifier(h)
        
        return h
    
    def backward_prel(self, prel, rule):
        prel = self.classifier.backward_prel(prel, rule)
        prel = self.act.backward_prel(prel, rule)
        prel = self.pre_classifier.backward_prel(prel, rule)
        prel = self.choose_token.backward_prel(prel, rule)
        prel = self.distilbert.backward_prel(prel, rule)
        
        return prel