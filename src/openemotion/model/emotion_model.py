import torch
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

from openemotion.config import CLASS_NAMES, IMG_SIZE


def create_model(num_classes: int = len(CLASS_NAMES)) -> nn.Module:
    """Create an EfficientNet-B0 model with customized output head."""
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )
    return model


def load_model(path: str, device: torch.device, num_classes: int = len(CLASS_NAMES)) -> nn.Module:
    """Load model weights from a .pth file."""
    model = create_model(num_classes)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


INFERENCE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
