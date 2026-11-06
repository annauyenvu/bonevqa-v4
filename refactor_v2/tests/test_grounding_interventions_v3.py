import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


def module():
    path = Path(__file__).parents[1] / "scripts" / "run_grounding_interventions_v3.py"
    spec = importlib.util.spec_from_file_location("grounding_v3", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_matched_nonlesion_preserves_box_geometry_and_avoids_lesion():
    mod = module()
    lesions = [[4, 4, 8, 7]]
    shifted = mod.matched_boxes(lesions, 20, 12)
    assert shifted[0][2] - shifted[0][0] == 4
    assert shifted[0][3] - shifted[0][1] == 3
    assert not np.logical_and(mod.box_mask(lesions, 20, 12), mod.box_mask(shifted, 20, 12)).any()


def test_background_replacement_keeps_lesion_pixels_only():
    mod = module()
    array = np.arange(12 * 20 * 3, dtype=np.uint8).reshape(12, 20, 3)
    image = Image.fromarray(array)
    lesions = [[4, 4, 8, 7]]
    changed = np.asarray(mod.intervene(image, lesions, "background_replacement"))
    lesion = mod.box_mask(lesions, 20, 12)
    assert np.array_equal(changed[lesion], array[lesion])
    assert not np.array_equal(changed[~lesion], array[~lesion])


def test_fractional_annotation_placebo_preserves_exact_raster_area():
    mod = module()
    lesions = [[4.3, 4.2, 8.8, 7.8]]
    shifted = mod.matched_boxes(lesions, 20, 12)
    original = mod.box_mask(lesions, 20, 12)
    placebo = mod.box_mask(shifted, 20, 12)
    assert original.sum() == placebo.sum()
    assert not (original & placebo).any()
