# Advancing Attribution-Based Explainability through Multi-Component Evaluation and Relative Absolute Magnitude Propagation

This repository contains the source code for the new __Relative Absolute Magnitude Propagation__ attribution method and the __Global Evaluation Metric__

## Relative Absolute Magnitude Propagation

A novel Layer-Wise Propagation rule, referred to as Relative Absolute Magnitude Propagation (RAMP). This rule effectively addresses the issue of incorrect relative attribution between neurons within the same layer that exhibit varying absolute magnitude activations. We apply this rule to three different, including the very recent Vision Transformer.

![Alt text](images/image-1.png)
*Figure 1. RAMP visualizations for VGG architecture - ImageNet*

![Alt text](images/image-2.png)
*Figure 2. RAMP visualizations for Vision Transformer architecture - PascalVOC*

## Global Evaluation Metric

A new evaluation method, Global Attribution Evaluation (GAE), which offers a novel perspective on evaluating faithfulness and robustness of an attribution method by utilizing gradient-based masking, while combining those results with a localization method to achieve a comprehensive evaluation of explanation quality in a single score.

![Alt text](images/image-4.png)
*Figure 3. Top and bottom 5 scoring images on GAE metric out of a randomly sampled 1024 images - RAMP VGG ImageNet*

### Usage Example
#### Relative Absolute Magnitude Propagation
Import the required libraries
```
import torch
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms as T
from PIL import Image
import timm
from ramp_gae.ramp.models import TimmVGG16, TimmResNet, TimmVisionTransformer
from ramp_gae.ramp.relevancy_methods import IntRelevancyMethod
from ramp_gae.utils import preprocess_pil_image, visualize_tensor_relevance_batch
```
Load a model from timm and wrap it inside the RAMP class
```
device = 'cuda'
# model = timm.create_model('vgg16', pretrained=True)
# ramp_model = TimmVGG16(model)
# model = timm.create_model('resnet50', pretrained=True)
# ramp_model = TimmResNet(model)
model = timm.create_model('vit_base_patch16_224', pretrained=True)
ramp_model = TimmVisionTransformer(model)
ramp_model.to(device)
ramp_model.eval()
is_vit = isinstance(ramp_model, TimmVisionTransformer)
relevancy_method = IntRelevancyMethod(ramp_model, rule='intline', relevancy_type='contrastive', device=device)
```
Load an inference image and preprocess
```
image = Image.open(image_path)
image = preprocess_pil_image(image, is_vit=is_vit)
```
Calculate contrastive relevance using RAMP and visualize
```
x = image.unsqueeze(0)
r, _, _ = relevancy_method.relevancy(x, choose_max=True)
visualize_tensor_relevance_batch(x, r, is_vit=is_vit)
```