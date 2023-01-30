# Semantic Segmentation 

Implementation of Deeplabv3+ with W&amp;B logging

### Several backbones available:
- ResNet (resnet50, resnet101)
- Swin-Transformer (swinT, swinS, swinB, swinL)
- ConvNeXt (convnextT, convnextS, convnextB, convnextL, convnextXL)

Note that swinB is the base model from [Swin Transformers](https://github.com/microsoft/Swin-Transformer) where instead swinT, swinS and swinL have a dimension and complexity of about 0.25x, 0.5x and 2x of swinB.
Note that swinT is comparable to resnet50 and swinS to resnet101 complexity-wise.
ConvNeXt if from [here](https://github.com/facebookresearch/ConvNeXt).

Pretrained models on ImageNet for the swin transformers backbones can be downloaded [here](https://nextcloud.in.tum.de/index.php/s/Y5oNtKBLwXLnL8m) and placed inside ```backbone_checkpoints``` in the main folder.

### Several learning rate schedulers available:
- step (stepLR)
- cosine (CosineAnnealingLR)
- poly (PolyLR)

