import torch
import torch.nn as nn
import timm
from dataset import NUM_CLASSES


MODEL_CONFIGS = {
    # CNN Models
    "vgg16": {"timm_name": "vgg16_bn", "input_size": 224},
    "vgg19": {"timm_name": "vgg19_bn", "input_size": 224},
    "mobilenetv2_100": {"timm_name": "mobilenetv2_100", "input_size": 224},
    "mobilenetv3_large_100": {"timm_name": "mobilenetv3_large_100", "input_size": 224},
    "inception_v3": {"timm_name": "inception_v3", "input_size": 299},
    "tf_efficientnetv2_s": {"timm_name": "tf_efficientnetv2_s", "input_size": 224},
    "tf_efficientnetv2_m": {"timm_name": "tf_efficientnetv2_m", "input_size": 224},
    "resnet18": {"timm_name": "resnet18", "input_size": 224},
    "resnet50": {"timm_name": "resnet50", "input_size": 224},
    "seresnet50": {"timm_name": "seresnet50", "input_size": 224},
    "densenet121": {"timm_name": "densenet121", "input_size": 224},
    "convnext_small": {"timm_name": "convnext_small.fb_in22k", "input_size": 224},
    "ghostnet_100": {"timm_name": "ghostnet_100", "input_size": 224},
    # Transformer Models
    "vit_tiny_patch16_224": {"timm_name": "vit_tiny_patch16_224.augreg_in21k", "input_size": 224},
    "vit_base_patch16_224": {"timm_name": "vit_base_patch16_224.augreg_in21k", "input_size": 224},
    "deit_small_patch16_224": {"timm_name": "deit_small_patch16_224", "input_size": 224},
    "swin_base_patch4_window7_224": {"timm_name": "swin_base_patch4_window7_224", "input_size": 224},
    "mobilevit_s": {"timm_name": "mobilevit_s", "input_size": 256},
    "cvt_13": {"timm_name": "coatnet_1_224", "input_size": 224}, # CoAtNet hybrid (conv + attention)
    "crossvit_9_240": {"timm_name": "crossvit_9_240", "input_size": 240},
}


def get_model(model_name, num_classes=NUM_CLASSES, pretrained=True):
    if model_name in MODEL_CONFIGS:
        cfg = MODEL_CONFIGS[model_name]
        try:
            try:
                model = timm.create_model(
                    cfg["timm_name"],
                    pretrained=pretrained,
                    num_classes=num_classes,
                    drop_rate=0.2,
                    drop_path_rate=0.15,
                )
            except TypeError:
                model = timm.create_model(
                    cfg["timm_name"],
                    pretrained=pretrained,
                    num_classes=num_classes,
                )
        except RuntimeError as e:
            if pretrained and "pretrained" in str(e).lower():
                print(f"Warning: No pretrained weights for {model_name}. Using random init.")
                return get_model(model_name, num_classes, pretrained=False)
            raise
        input_size = cfg["input_size"]
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    return model, input_size


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
