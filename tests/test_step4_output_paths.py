"""Step 4 updates the configured mosaics and uses the configured patch folder."""
import json
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window


@pytest.mark.parametrize("custom,relative", [
    ("none", False), ("mosaics", False), ("patches", False),
    ("all", False), ("all", True),
])
def test_main_uses_configured_output_paths(step4, tmp_path, monkeypatch, custom, relative):
    work = tmp_path / "working folder"
    data = tmp_path / "configured images"
    work.mkdir()
    data.mkdir()
    (work / "masks").mkdir()
    (work / "cldDatabase").mkdir()
    candidates = data / "candidates"
    candidates.mkdir()
    pan_scenes = data / "pan"
    pan_scenes.mkdir()
    patches = data / "cloud patches" if custom in ("all", "patches") else work / "cldDatabase"
    patches.mkdir(exist_ok=True)
    custom_mosaics = custom in ("all", "mosaics")
    xs_path = data / "selected_xs.tif" if custom_mosaics else work / "mosaic_xs.tif"
    cloud_path = data / "selected_clouds.tif" if custom_mosaics else work / "mosaic_cloudmask.tif"
    pan_path = data / "selected_pan.tif" if custom_mosaics else work / "mosaic_pan.tif"

    transform = from_origin(1000, 10000, 60, 60)
    clean = np.random.default_rng(14).integers(50, 180, (4, 64, 64), dtype=np.uint8)
    mosaic = clean.copy()
    mosaic[:, 20:30, 20:30] = 250
    clouds = np.zeros((64, 64), np.uint8)
    clouds[20:30, 20:30] = 1
    pan = mosaic[0].copy()

    def write_raster(path, values):
        if values.ndim == 2:
            values = values[None]
        with rasterio.open(path, "w", driver="GTiff", width=64, height=64,
                           count=values.shape[0], dtype=values.dtype,
                           crs="EPSG:3826", transform=transform) as dst:
            dst.write(values)

    write_raster(xs_path, mosaic)
    write_raster(cloud_path, clouds)
    write_raster(pan_path, pan)
    write_raster(candidates / "A0000001.img", clean)
    write_raster(pan_scenes / "A0000002.tif", clean[0])
    write_raster(work / "masks" / "A0000001.tif", np.ones((64, 64), np.uint8))
    write_raster(data / "land.tif", np.ones((64, 64), np.uint8))

    decoys = {}
    if custom_mosaics:
        for name, values in (("mosaic_xs.tif", np.full_like(clean, 99)),
                             ("mosaic_cloudmask.tif", np.full_like(clouds, 77)),
                             ("mosaic_pan.tif", np.full_like(pan, 88))):
            path = work / name
            write_raster(path, values)
            decoys[path] = values.copy()
    if patches != work / "cldDatabase":
        # A same-named folder in the working directory must not receive patches.
        (work / "cldDatabase" / "keep.txt").write_text("unrelated")

    def configured(path):
        return os.path.relpath(path, work) if relative else str(path)

    (work / "para_cld.txt").write_text(json.dumps({
        "res": 60, "pan_res": 60, "trueBits": 8,
        "CandiImgPath": configured(candidates), "CldMaskPath": configured(patches),
        "MosaicImgAddress": configured(xs_path), "MosaicCldAddress": configured(cloud_path),
        "PanMosaicImgAddress": configured(pan_path), "PanImgAddress": configured(pan_scenes),
        "CandiMaskFolderName": "masks", "seaBorderAddress": configured(data / "land.tif"),
    }))
    (work / "ProcessOrder.txt").write_text("primary_a.img,primary_b.img")
    windows = np.empty(1, dtype=object)
    windows[0] = Window(0, 0, 64, 64)
    np.save(work / "allWindow.npy", windows)
    np.save(work / "transform.npy", [[1000, 10000], [1000, 10000]])
    monkeypatch.chdir(work)
    monkeypatch.setattr(step4, "listdir", lambda path: sorted(os.listdir(path)))

    step4.main()

    with rasterio.open(cloud_path) as result:
        assert np.all(result.read(1)[20:30, 20:30] == 0)
    with rasterio.open(xs_path) as result:
        assert np.any(result.read()[:, 20:30, 20:30] != mosaic[:, 20:30, 20:30])
    with rasterio.open(pan_path) as result:
        assert np.any(result.read(1)[20:30, 20:30] != pan[20:30, 20:30])
    for path, values in decoys.items():
        with rasterio.open(path) as result:
            actual = result.read() if values.ndim == 3 else result.read(1)
            np.testing.assert_array_equal(actual, values)
    assert list(patches.iterdir()) == []
    if patches != work / "cldDatabase":
        assert [path.name for path in (work / "cldDatabase").iterdir()] == ["keep.txt"]
