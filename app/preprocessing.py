"""
app/preprocessing.py — the single source of truth for everything between
"raw image bytes" and "feature vector". Imported by both scripts/train_head.py
and app/classifier.py. There is no second copy of this logic anywhere.
"""

import io

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# 1. Image -> tensor
# All three candidate backbones share the same ImageNet input contract:
# 224x224 pixels, ImageNet channel means/stds.
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    # Explicit (224, 224), not the usual Resize(256)+CenterCrop(224): tiles are
    # already small crops, so we want the whole tile seen — deterministically,
    # identically, in both training and serving.
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_image(data: bytes) -> Image.Image:
    """Raw file bytes -> RGB PIL image. Raises ValueError for bytes that
    aren't a decodable image — the endpoint converts that to HTTP 400."""
    try:
        img = Image.open(io.BytesIO(data))
        return img.convert("RGB")  # survives grayscale / RGBA / palette tiles
    except Exception as exc:
        raise ValueError(f"input is not a decodable image: {exc}") from exc


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL image -> (3, 224, 224). Training path: stack these into batches."""
    return _transform(img)


def preprocess(img: Image.Image) -> torch.Tensor:
    """PIL image -> (1, 3, 224, 224). Serving path: one image at a time."""
    return _transform(img).unsqueeze(0)


# ---------------------------------------------------------------------------
# 2. Backbone -> feature extractor
# Chopping the classification layer off is architecture-specific, so each
# backbone gets its own tiny adapter, selected from one registry below —
# same single-source-of-truth principle as the shared resize/normalize logic.
# ---------------------------------------------------------------------------


def _adapt_resnet18(model: nn.Module) -> nn.Module:
    # children() = [conv1, bn1, relu, maxpool, layer1..4, avgpool, fc].
    # Dropping fc keeps avgpool -> output (N, 512, 1, 1).
    return nn.Sequential(*list(model.children())[:-1])


class _MobilenetV2Features(nn.Module):
    """MobileNetV2's spatial pooling happens inside its forward(), not in a
    module — so children()[:-1] alone would leave a (N, 1280, 7, 7) tensor.
    This wrapper mirrors what its own forward() does before the classifier:
    features() -> AdaptiveAvgPool2d(1) -> (N, 1280, 1, 1)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x))


def _adapt_squeezenet1_1(model: nn.Module) -> nn.Module:
    # classifier = [Dropout, Conv2d(512->1000, 1x1), ReLU, AdaptiveAvgPool2d].
    # No clean (pool, fc) pair to chop, so replace the 1x1 conv with Identity:
    # the module's existing avgpool then yields (N, 512, 1, 1). Safe to mutate
    # in place because build_feature_extractor() always constructs a fresh model.
    model.classifier[1] = nn.Identity()
    return model


BACKBONE_BUILDERS = {
    "resnet18": models.resnet18,
    "mobilenet_v2": models.mobilenet_v2,
    "squeezenet1_1": models.squeezenet1_1,
}

FEATURE_ADAPTERS = {
    "resnet18": _adapt_resnet18,
    "mobilenet_v2": _MobilenetV2Features,
    "squeezenet1_1": _adapt_squeezenet1_1,
}

FEATURE_DIMS = {"resnet18": 512, "mobilenet_v2": 1280, "squeezenet1_1": 512}


def build_feature_extractor(backbone_name: str) -> nn.Module:
    """Fresh ImageNet-pretrained backbone -> frozen feature-extraction module.
    Flattened output = the feature vector."""
    if backbone_name not in BACKBONE_BUILDERS:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. Known: {sorted(BACKBONE_BUILDERS)}"
        )
    extractor = FEATURE_ADAPTERS[backbone_name](
        BACKBONE_BUILDERS[backbone_name](weights="DEFAULT")
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad_(False)
    return extractor


def extract_features(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """(N, 3, 224, 224) -> (N, D). Batched + gradient-free, per the plan."""
    with torch.no_grad():
        out = model(batch)
    return torch.flatten(out, 1)


if __name__ == "__main__":
    # Smoke test: build each backbone, push one image through, check dims.
    # NOTE: first run downloads the three pretrained weight files (~65 MB total,
    # one time only) — this is the plan's expected one-time setup download.
    for name in BACKBONE_BUILDERS:
        feats = extract_features(
            build_feature_extractor(name), torch.zeros(1, 3, 224, 224)
        )
        assert feats.shape == (1, FEATURE_DIMS[name]), (name, feats.shape)
        print(f"{name}: OK, feature dim = {feats.shape[1]}")