from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHeadV3Plus, DeepLabV3
from .backbone import resnet
from .backbone.swin_transformer import SwinB, SwinT, SwinS, SwinL
from .backbone.convnext import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge

def _segm_resnet(backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    return DeepLabV3(backbone, classifier)

###########################

def _segm_swinT(num_classes, pretrained_backbone):
    inplanes = 768
    low_level_planes = 96
    backbone = SwinT(pretrained=pretrained_backbone)
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate=[6, 12, 18])
    return DeepLabV3(backbone, classifier)

def _segm_swinS(num_classes, pretrained_backbone):
    inplanes = 768
    low_level_planes = 96
    backbone = SwinS(pretrained=pretrained_backbone)
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate=[6, 12, 18])
    return DeepLabV3(backbone, classifier)

def _segm_swinB(num_classes, pretrained_backbone):
    inplanes = 1024
    low_level_planes = 128
    backbone = SwinB(pretrained=pretrained_backbone)
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate=[6, 12, 18])
    return DeepLabV3(backbone, classifier)

def _segm_swinL(num_classes, pretrained_backbone):
    inplanes = 1536
    low_level_planes = 192
    backbone = SwinL(pretrained=pretrained_backbone)
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate=[6, 12, 18])
    return DeepLabV3(backbone, classifier)

###########################

def _segm_convnextT(num_classes, pretrained_backbone):
    inplanes = 768
    low_level_planes = 96
    backbone = convnext_tiny(pretrained=pretrained_backbone)
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate=[6, 12, 18])
    return DeepLabV3(backbone, classifier)

def _segm_convnextS(num_classes, pretrained_backbone):
    inplanes = 768
    low_level_planes = 96
    backbone = convnext_small(pretrained=pretrained_backbone)
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate=[6, 12, 18])
    return DeepLabV3(backbone, classifier)

def _segm_convnextB(num_classes, pretrained_backbone):
    inplanes = 1024
    low_level_planes = 128
    backbone = convnext_base(pretrained=pretrained_backbone)
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate=[6, 12, 18])
    return DeepLabV3(backbone, classifier)


def _segm_convnextL(num_classes, pretrained_backbone):
    inplanes = 1536
    low_level_planes = 192
    backbone = convnext_large(pretrained=pretrained_backbone)
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate=[6, 12, 18])
    return DeepLabV3(backbone, classifier)


def _segm_convnextXL(num_classes, pretrained_backbone):
    inplanes = 2048
    low_level_planes = 256
    backbone = convnext_xlarge(pretrained=pretrained_backbone)
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate=[6, 12, 18])
    return DeepLabV3(backbone, classifier)





def _load_model(backbone, num_classes, output_stride=8, pretrained_backbone=True):

    if backbone.startswith('resnet'):
        model = _segm_resnet(backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

    elif backbone.startswith('swinT'):
        model = _segm_swinT(num_classes, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('swinS'):
        model = _segm_swinS(num_classes, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('swinB'):
        model = _segm_swinB(num_classes, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('swinL'):
        model = _segm_swinL(num_classes, pretrained_backbone=pretrained_backbone)   

    elif backbone.startswith('convnextT'):
        model = _segm_convnextT(num_classes, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('convnextS'):
        model = _segm_convnextS(num_classes, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('convnextB'):
        model = _segm_convnextB(num_classes, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('convnextL'):
        model = _segm_convnextL(num_classes, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('convnextXL'):
        model = _segm_convnextXL(num_classes, pretrained_backbone=pretrained_backbone)

    else:
        raise NotImplementedError
    return model


## RESNET

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


## Swin-Transformer

def deeplabv3plus_swinT(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a Swin backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('swinT', num_classes, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_swinS(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a Swin backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('swinS', num_classes, pretrained_backbone=pretrained_backbone)
    
def deeplabv3plus_swinB(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a Swin backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('swinB', num_classes, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_swinL(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a Swin backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('swinL', num_classes, pretrained_backbone=pretrained_backbone)


## ConvNeXt


def deeplabv3plus_convnextT(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ConvNeXt backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('convnextT', num_classes, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_convnextS(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ConvNeXt backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('convnextS', num_classes, pretrained_backbone=pretrained_backbone)
    
def deeplabv3plus_convnextB(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ConvNeXt backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('convnextB', num_classes, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_convnextL(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ConvNeXt backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('convnextL', num_classes, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_convnextXL(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ConvNeXt backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('convnextXL', num_classes, pretrained_backbone=pretrained_backbone)

