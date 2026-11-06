from PIL import Image

from src.preprocessing import canonicalize_boxes, canonicalize_raster_and_boxes


def test_exif_orientation_6_rotates_boxes_and_size():
    image = Image.new("RGB", (100, 60))
    exif = image.getexif()
    exif[274] = 6
    image.info["exif"] = exif.tobytes()
    canonical, boxes = canonicalize_raster_and_boxes(image, [[10, 5, 30, 25]])
    assert canonical.size == (60, 100)
    assert boxes == [[35.0, 10.0, 55.0, 30.0]]


def test_all_exif_box_transforms_stay_in_canonical_bounds():
    raw_size = (100, 60)
    for orientation in range(1, 9):
        box = canonicalize_boxes([[5, 7, 90, 50]], raw_size, orientation)[0]
        canonical_size = (60, 100) if orientation in (5, 6, 7, 8) else raw_size
        assert 0 <= box[0] < box[2] <= canonical_size[0]
        assert 0 <= box[1] < box[3] <= canonical_size[1]
