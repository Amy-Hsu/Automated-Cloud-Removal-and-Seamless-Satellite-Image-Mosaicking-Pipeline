"""The cloud-tolerance denominator includes zero-valued clear pixels."""
import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window


@pytest.fixture
def tolerance_run(tmp_path, monkeypatch, step4):
    import os

    transform = from_origin(1000, 10000, 60, 60)
    mosaic = np.random.default_rng(15).integers(50, 180, (4, 64, 64), dtype=np.uint8)
    for folder in ("candidates", "cldDatabase", "masks", "pan"):
        (tmp_path / folder).mkdir()

    def write_raster(path, data, raster_transform=transform):
        if data.ndim == 2:
            data = data[None]
        with rasterio.open(tmp_path / path, "w", driver="GTiff", height=data.shape[1],
                           width=data.shape[2], count=data.shape[0], dtype=data.dtype,
                           crs="EPSG:3826", transform=raster_transform) as dst:
            dst.write(data)

    write_raster("mosaic_xs.tif", mosaic)
    write_raster("mosaic_pan.tif", mosaic[0])
    write_raster("land.tif", np.ones((64, 64), np.uint8))
    cloudmask = np.zeros((64, 64), np.uint8)
    cloudmask[20:30, 20:30] = 1
    write_raster("mosaic_cloudmask.tif", cloudmask)
    write_raster("candidates/A0000001.img", mosaic)
    windows = np.empty(1, dtype=object)
    windows[0] = Window(0, 0, 64, 64)
    np.save(tmp_path / "allWindow.npy", windows)
    np.save(tmp_path / "transform.npy", [[1000, 10000], [1000, 10000]])
    (tmp_path / "ProcessOrder.txt").write_text("primary_a.img,primary_b.img")
    (tmp_path / "para_cld.txt").write_text(json.dumps({
        "res": 60, "pan_res": 60, "trueBits": 8,
        "CandiImgPath": str(tmp_path / "candidates"),
        "CldMaskPath": str(tmp_path / "cldDatabase"),
        "MosaicImgAddress": str(tmp_path / "mosaic_xs.tif"),
        "MosaicCldAddress": str(tmp_path / "mosaic_cloudmask.tif"),
        "PanMosaicImgAddress": str(tmp_path / "mosaic_pan.tif"),
        "PanImgAddress": str(tmp_path / "pan"),
        "CandiMaskFolderName": "masks", "seaBorderAddress": str(tmp_path / "land.tif"),
    }))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(step4, "listdir", lambda path: sorted(os.listdir(path)))
    return SimpleNamespace(root=tmp_path, mosaic=mosaic, write=write_raster)


@pytest.mark.parametrize("case,accepted", [
    ("zero_clear", True), ("one_clear", True), ("half_cloudy", False),
    ("one_fifth_cloudy", True), ("all_cloudy", False), ("no_cloud_overlap", False),
])
def test_cloud_tolerance_counts_every_patch_pixel(step4, tolerance_run, monkeypatch, case, accepted):
    run = tolerance_run
    cloud_codes = np.ones((64, 64), np.uint8)
    if case == "zero_clear":
        cloud_codes[:] = 0
    elif case == "half_cloudy":
        cloud_codes[:25] = 0
        cloud_codes[25:] = 10
    elif case == "one_fifth_cloudy":
        cloud_codes[:] = 0
        cloud_codes[::5] = 10
    elif case == "all_cloudy":
        cloud_codes[:] = 10
    elif case == "no_cloud_overlap":
        # A seven-row SSIM overlap at the patch margin contains no cloud pixels.
        run.write("candidates/A0000001.img", run.mosaic[:, :15])
    run.write("masks/A0000001.tif", cloud_codes)
    run.write("pan/A0000002.tif", run.mosaic[0])
    clones = []
    clone = step4.seamless_clone_over_nodata

    def record_clone(*args, **kwargs):
        clones.append(True)
        return clone(*args, **kwargs)

    monkeypatch.setattr(step4, "seamless_clone_over_nodata", record_clone)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        step4.main()
    assert bool(clones) is accepted
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    if not accepted:
        with rasterio.open(run.root / "mosaic_cloudmask.tif") as result:
            assert np.all(result.read(1)[20:30, 20:30] == 1)
