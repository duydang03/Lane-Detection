from .unet import UNet
from .deeplab import DeepLabV3Plus
from .espnet import ESPNet
from .twin import TwinLiteNet

def build_model(model_name, num_classes=1):

    models = {
        "unet": UNet,
        "deeplab": DeepLabV3Plus,
        "espnet": ESPNet,
        "twin": TwinLiteNet
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    return models[model_name](num_classes=num_classes)