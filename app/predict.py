
import io, json
from pathlib import Path
from typing import List, Tuple
import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
# --- add near the top of predict.py ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path(__file__).with_name("ldh_classifier.pt")
LABELS_PATH = Path(__file__).with_name("labels.json")

_model = None
_labels = None
_tfms = None

def _load():
    global _model, _labels, _tfms
    if _model is None:
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        _labels = ckpt.get("class_names", json.loads(LABELS_PATH.read_text()))
        m = models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Linear(m.fc.in_features, len(_labels))
        m.load_state_dict(ckpt["state_dict"], strict=True)
        m.eval()
        _model = m.to(DEVICE)

        img_size = ckpt.get("img_size", 224)
        mean = ckpt.get("normalize", {}).get("mean", [0.5])
        std = ckpt.get("normalize", {}).get("std", [0.5])
        _tfms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return _model, _labels, _tfms

def predict_image(img_bytes: bytes, topk: int = 2) -> List[Tuple[str, float]]:
    model, labels, tfms = _load()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = tfms(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].cpu()
    vals, idxs = torch.topk(probs, k=min(topk, len(labels)))
    return [(labels[i], float(vals[j])) for j, i in enumerate(idxs.tolist())]

# in app/predict.py
def get_labels():
    _, labels, _ = _load()
    return labels

def gradcam_on_image(img_bytes: bytes, target_class: int | None = None):
    """
    Returns:
      overlay_rgb: np.ndarray (H,W,3) uint8
      pred_label:  str
      pred_prob:   float
      topk:        list[(label, prob)]
    """
    import io, numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model, labels, tfms = _load()

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = tfms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu()
    topk_vals, topk_idxs = torch.topk(probs, k=min(5, len(labels)))
    topk = [(labels[i], float(topk_vals[j])) for j, i in enumerate(topk_idxs.tolist())]
    pred_idx = int(topk_idxs[0]) if target_class is None else int(target_class)
    pred_label, pred_prob = labels[pred_idx], float(probs[pred_idx])

    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])  # no use_cuda arg
    grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred_idx)])[0]

    # prepare overlay base (ensure 3-channel for visualization)
    base = np.array(img.resize((grayscale_cam.shape[1], grayscale_cam.shape[0])), dtype=np.float32)/255.0
    if base.ndim == 2:
        base = np.stack([base]*3, axis=-1)

    overlay_rgb = show_cam_on_image(base, grayscale_cam, use_rgb=True)  # uint8
    return overlay_rgb, pred_label, pred_prob, topk

