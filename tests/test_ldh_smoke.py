from app.predict import predict_image
from PIL import Image
import io

def test_predict_on_dummy_image():
    img = Image.new("RGB", (224,224), color=(128,128,128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    preds = predict_image(buf.getvalue(), topk=2)
    assert len(preds) >= 1
