"""Single raster preprocessing contract shared by train, eval, and app."""

from PIL import Image, ImageOps


RASTER_PREPROCESSING_VERSION = "bonevqa-exif-canonical-v3"


def exif_orientation(image: Image.Image) -> int:
    return int(image.getexif().get(274, 1) or 1)


def canonicalize_boxes(boxes, raw_size, orientation: int):
    """Transform raw-file box coordinates exactly as ``exif_transpose``.

    Coordinates are continuous edge coordinates in ``[0, width] x
    [0, height]``.  The four corners are transformed and then re-enclosed,
    making the function valid for every EXIF orientation 1--8.
    """

    if boxes is None:
        return None
    width, height = raw_size

    def point(x, y):
        transforms = {
            1: (x, y),
            2: (width - x, y),
            3: (width - x, height - y),
            4: (x, height - y),
            5: (y, x),
            6: (height - y, x),
            7: (height - y, width - x),
            8: (y, width - x),
        }
        return transforms.get(int(orientation), (x, y))

    transformed = []
    for box in boxes:
        x1, y1, x2, y2 = map(float, box[:4])
        corners = [point(x1, y1), point(x1, y2), point(x2, y1), point(x2, y2)]
        xs, ys = zip(*corners)
        transformed.append([min(xs), min(ys), max(xs), max(ys), *box[4:]])
    return transformed


def canonicalize_raster(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def canonicalize_raster_and_boxes(image: Image.Image, boxes):
    orientation = exif_orientation(image)
    raw_size = image.size
    return canonicalize_raster(image), canonicalize_boxes(boxes, raw_size, orientation)


def letterbox(image: Image.Image, size: int, fill=(0, 0, 0)) -> Image.Image:
    image = canonicalize_raster(image)
    w, h = image.size
    scale = min(size / max(w, 1), size / max(h, 1))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas
