from PIL import Image


class ResizeMaxSide:
    def __init__(self, max_side=512):
        self.max_side = max_side

    def __call__(self, image):
        w, h = image.size
        scale = self.max_side / float(max(w, h))
        if scale >= 1.0:
            return image
        return image.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.BILINEAR)


class ToRGB:
    def __call__(self, image):
        return image.convert("RGB")


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image
