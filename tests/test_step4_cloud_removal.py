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


def test_patch_margins_match_patch_near_window_edge(step4, tmp_path):
    """A cloud patch touching the window edge must be written at its true size (#2)."""
    import warnings

    import rasterio
    from rasterio import Affine

    labels = np.zeros((100, 120), np.uint8)
    for x, y, w, h in [(3, 40, 20, 15), (50, 2, 10, 30), (100, 80, 16, 15), (40, 40, 20, 20)]:
        labels[:] = 0
        labels[y:y + h, x:x + w] = 1
        expdx, expdy, expdw, expdh, height, width = step4.patch_margins(x, y, w, h, labels.shape)
        assert min(expdx, expdy, expdw, expdh) >= 0
        patch = labels[y - expdy:y + h + expdh, x - expdx:x + w + expdw]
        assert patch.shape == (height, width)

        path = tmp_path / f"{x}_{y}_cld.img"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(path, "w+", height=height, width=width, count=1,
                               dtype=rasterio.uint8, crs="EPSG:3826",
                               transform=Affine.translation(0, 1000) * Affine.scale(6, -6)) as dst:
                dst.write(patch, 1)
            with rasterio.open(path) as src:
                np.testing.assert_array_equal(src.read(1), patch)


def test_seamless_clone_ignores_mosaic_nodata(step4):
    """Mosaic no-data next to a cloud patch must not darken the Poisson fill (#4).

    In the Sentinel-2 demo, patches at the edge of the mosaic footprint were
    filled with 0-valued speckles because the 0 no-data pixels acted as the
    boundary of the Poisson solve.
    """
    import cv2
    from scipy import ndimage

    rng = np.random.default_rng(4)

    def texture():  # dark land, as 12-bit data divided by 16
        t = ndimage.gaussian_filter(rng.normal(0, 1, (120, 120)), 3) * 60 + 25
        return np.clip(t, 1, 255).astype(np.uint8)[..., None].repeat(3, 2)

    src, dst = texture(), texture()
    nodata = np.zeros((120, 120), bool)
    nodata[:, :50] = True              # outside the mosaic footprint
    dst[nodata] = 0
    mask = np.zeros((120, 120), np.uint8)
    mask[30:90, 42:100] = 255          # cloud patch next to the footprint edge
    inside = mask == 255

    plain = cv2.seamlessClone(src, dst, mask, (71, 60), cv2.NORMAL_CLONE)
    assert np.any(plain[inside] == 0)  # the speckles reported in #4

    out = step4.seamless_clone_over_nodata(src, dst, mask, (71, 60), nodata)
    assert not np.any(out[inside] == 0)
    assert abs(out[inside].mean() - src[inside].mean()) < 3
    np.testing.assert_array_equal(out[~inside & ~nodata], dst[~inside & ~nodata])
