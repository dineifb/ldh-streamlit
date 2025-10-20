
import io, json
from pathlib import Path
from typing import List, Tuple
import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

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
