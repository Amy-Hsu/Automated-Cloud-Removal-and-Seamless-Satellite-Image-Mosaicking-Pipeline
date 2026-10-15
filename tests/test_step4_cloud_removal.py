"""Unit tests for Step 4 helpers."""
import numpy as np


def test_border_elems_generic(step4):
    a = np.arange(25).reshape(5, 5)
    border = step4.border_elems_generic(a, 1)
    assert sorted(border) == sorted(
        set(a[0]) | set(a[-1]) | set(a[:, 0]) | set(a[:, -1])
    )
    assert len(step4.border_elems_generic(a, 2)) == 24  # all but the centre


def test_seamless_clone_uses_source_texture():
    """Step 4's Poisson fill relies on cv2.seamlessClone copying the source texture.

    OpenCV 4.11+ keeps the destination texture instead, which silently leaves
    clouds in place; requirements.txt pins OpenCV below 4.11 for this reason.
    """
    import cv2
    from scipy import ndimage

    rng = np.random.default_rng(1)

    def texture():
        t = ndimage.gaussian_filter(rng.normal(0, 1, (300, 300)), 3) * 400 + 120
        return np.clip(t, 0, 255).astype(np.uint8)[..., None].repeat(3, 2)

    def highpass(x):
        x = x.astype(float)
        return x - ndimage.uniform_filter(x, (7, 7, 1))

    dst, src = texture(), texture()
    mask = np.zeros((300, 300), np.uint8)
    mask[100:200, 100:200] = 255
    out = cv2.seamlessClone(src, dst, mask, (150, 150), cv2.NORMAL_CLONE)
    inside = mask == 255
    corr_src = np.corrcoef(highpass(out)[inside].ravel(), highpass(src)[inside].ravel())[0, 1]
    corr_dst = np.corrcoef(highpass(out)[inside].ravel(), highpass(dst)[inside].ravel())[0, 1]
    assert corr_src > 0.7 and corr_dst < 0.3, f"OpenCV {cv2.__version__} seamlessClone is unsuitable"


def test_pan_scene_path_prefers_spot_layout_and_ignores_case(step4, tmp_path):
    import os

    (tmp_path / "a0000002.tif").write_bytes(b"")
    found = step4.pan_scene_path(str(tmp_path), "A0000002")
    assert os.path.isfile(found) and os.path.basename(found).lower() == "a0000002.tif"
    spot = tmp_path / "A0000004" / "A0000004_B1.DAT.raw"
    spot.parent.mkdir()
    spot.write_bytes(b"")
    found = step4.pan_scene_path(str(tmp_path), "a0000004")
    assert os.path.isfile(found) and found.lower() == str(spot).lower()
