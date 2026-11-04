"""Unit tests for Step 4 helpers."""
import numpy as np
import pytest


@pytest.fixture
def scene_pool(tmp_path):
    """Small real rasters shared by direct scoring and main-flow tests."""
    import json
    from types import SimpleNamespace

    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    transform = from_origin(1000, 10000, 60, 60)
    mosaic = np.random.default_rng(16).integers(1, 255, (4, 64, 64), dtype=np.uint8)

    def write_raster(path, data, raster_transform=transform):
        if data.ndim == 2:
            data = data[None]
        with rasterio.open(
            path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
            count=data.shape[0], dtype=data.dtype, crs="EPSG:3826",
            transform=raster_transform,
        ) as dst:
            dst.write(data)

    def create(candidate_count, patch_count):
        candidates = tmp_path / "candidates"
        patches = tmp_path / "patches"
        candidates.mkdir()
        patches.mkdir()
        (tmp_path / "cldDatabase").mkdir()
        candidate_files = [f"scene_{index:03}.img" for index in range(candidate_count)]
        for index, name in enumerate(candidate_files):
            data = mosaic if index == candidate_count - 1 else np.full_like(mosaic, 100)
            write_raster(candidates / name, data)
        write_raster(tmp_path / "xs.tif", mosaic)
        write_raster(tmp_path / "pan.tif", mosaic[0])
        write_raster(tmp_path / "land.tif", np.ones((64, 64), np.uint8))
        clouds = np.zeros((64, 64), np.uint8)
        cloud_files = []
        for index in range(patch_count):
            patch = np.zeros_like(clouds)
            start = 10 + 30 * index
            patch[start:start + 10, start:start + 10] = 1
            clouds |= patch
            name = f"{index}_cld.img"
            write_raster(patches / name, patch)
            cloud_files.append(name)
        write_raster(tmp_path / "clouds.tif", clouds)

        windows = np.empty(1, dtype=object)
        windows[0] = Window(0, 0, 64, 64)
        np.save(tmp_path / "allWindow.npy", windows)
        np.save(tmp_path / "transform.npy", [[1000, 10000], [1000, 10000]])
        (tmp_path / "ProcessOrder.txt").write_text("primary_a.img,primary_b.img")
        (tmp_path / "para_cld.txt").write_text(json.dumps({
            "res": 60, "pan_res": 60, "trueBits": 8,
            "CandiImgPath": str(candidates),
            "CldMaskPath": str(tmp_path / "cldDatabase"),
            "MosaicImgAddress": str(tmp_path / "xs.tif"),
            "MosaicCldAddress": str(tmp_path / "clouds.tif"),
            "PanMosaicImgAddress": str(tmp_path / "pan.tif"),
            "PanImgAddress": str(tmp_path),
            "CandiMaskFolderName": "masks",
            "seaBorderAddress": str(tmp_path / "land.tif"),
            # Exercise real scoring/ranking without entering the unrelated fill stage.
            "ssim_threshold": 1.1,
        }))
        return SimpleNamespace(root=tmp_path, candidates=candidates, patches=patches,
                               candidate_files=candidate_files, cloud_files=cloud_files,
                               mosaic=mosaic, write_raster=write_raster)

    return create


def score_pool(step4, pool, excluded_scenes=()):
    import rasterio

    with rasterio.open(pool.root / "xs.tif") as mosaic, rasterio.open(pool.root / "land.tif") as land:
        scores = step4.score_candidates(pool.candidate_files, pool.cloud_files,
                                        pool.candidates, pool.patches, mosaic, land, 60, 1,
                                        excluded_scenes)
        assert not mosaic.closed and not land.closed
        return scores


@pytest.mark.parametrize("candidate_count", [0, 1, 49, 50, 51, 73])
@pytest.mark.parametrize("patch_count", [1, 2])
def test_candidate_scores_cover_the_actual_scene_pool(step4, scene_pool, candidate_count, patch_count):
    """Score every candidate directly, including scene 51 and beyond (#16)."""
    scores = score_pool(step4, scene_pool(candidate_count, patch_count))
    assert scores.shape == (candidate_count, patch_count)
    if candidate_count:
        np.testing.assert_allclose(scores[-1], 1)
        assert np.all(scores.argmax(axis=0) == candidate_count - 1)


def test_candidate_scores_without_cloud_patches(step4, scene_pool):
    assert score_pool(step4, scene_pool(1, 0)).shape == (1, 0)


@pytest.mark.parametrize("case,expected", [
    ("zero", -3), ("sea", -3), ("no_overlap", -1), ("missing_bands", -1),
    ("excluded", -2), ("excluded_zero", -2), ("partial_overlap", 0.95),
])
def test_candidate_score_rules(step4, scene_pool, case, expected):
    """Keep the legacy penalty, rejection values and exclusion precedence."""
    from rasterio.transform import from_origin

    pool = scene_pool(1, 1)
    candidate = pool.candidates / pool.candidate_files[0]
    excluded = ()
    if case in ("zero", "excluded_zero"):
        pool.write_raster(candidate, np.zeros_like(pool.mosaic))
    elif case == "sea":
        pool.write_raster(pool.root / "land.tif", np.zeros((64, 64), np.uint8))
    elif case == "no_overlap":
        pool.write_raster(candidate, pool.mosaic, from_origin(100000, 10000, 60, 60))
    elif case == "missing_bands":
        pool.write_raster(candidate, pool.mosaic[0])
    elif case == "partial_overlap":
        pool.write_raster(candidate, pool.mosaic[:, :, :48])
    if case in ("excluded", "excluded_zero"):
        # ProcessOrder contains serialized scene lists, not just bare filenames.
        excluded = ("['scene_000.img']",)
    assert score_pool(step4, pool, excluded)[0, 0] == pytest.approx(expected)


@pytest.mark.parametrize("window_count", [1, 2])
def test_main_scores_more_than_fifty_candidates(step4, scene_pool, monkeypatch, window_count):
    """Keep a real main-flow regression without changing NumPy globally."""
    import os

    pool = scene_pool(73, 2)
    if window_count == 2:
        windows = np.load(pool.root / "allWindow.npy", allow_pickle=True)
        np.save(pool.root / "allWindow.npy", np.tile(windows, 2))
        np.save(pool.root / "transform.npy", [[1000, 10000]] * 3)
        (pool.root / "ProcessOrder.txt").write_text(
            "['scene_000.img'],['scene_001.img'],['scene_002.img']"
        )
    captured = []
    score_candidates = step4.score_candidates

    def capture_scores(*args, **kwargs):
        scores = score_candidates(*args, **kwargs)
        captured.append(scores.copy())
        return scores

    monkeypatch.chdir(pool.root)
    monkeypatch.setattr(step4, "listdir", lambda path: sorted(os.listdir(path)))
    monkeypatch.setattr(step4, "score_candidates", capture_scores)
    step4.main()
    assert len(captured) == window_count
    for scores in captured:
        assert scores.shape == (73, 2)
        assert np.all(scores[-1] >= 0.95)
        assert np.all(scores.argmax(axis=0) == 72)
    if window_count == 2:
        # First window excludes the two original mosaic scenes; subsequent
        # windows exclude only the newly added scene.
        np.testing.assert_array_equal(captured[0][:2], -2)
        assert np.all(captured[0][2] != -2)
        np.testing.assert_array_equal(captured[1][2], -2)
        assert np.all(captured[1][:2] != -2)


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
