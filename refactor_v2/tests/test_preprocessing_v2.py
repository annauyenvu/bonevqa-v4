from PIL import Image

from src.preprocessing import letterbox


def test_letterbox_preserves_geometry():
    image = Image.new("RGB", (400, 100), "white")
    output = letterbox(image, 224)
    assert output.size == (224, 224)
    # 4:1 image becomes 224x56 rather than being stretched to a square.
    assert output.getpixel((112, 84)) == (255, 255, 255)
    assert output.getpixel((112, 20)) == (0, 0, 0)
