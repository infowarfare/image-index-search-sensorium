import base64
import io
from PIL import Image

def resize_image(base64_image: str, max_size: int = 512) -> str:
    """Reduce image size (and possibly image quality) and return as Base64 encoded string."""
    image_bytes = base64.b64decode(base64_image)
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail((max_size, max_size))  # behält Seitenverhältnis
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")