"""Cloud-free windows must not prevent later windows from being filled."""
import json
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window


@pytest.mark.parametrize("case", ["clear", "small_scrap", "cloud_in_second_window"])
def test_main_skips_windows_without_fillable_cloud_patches(step4, tmp_path, monkeypatch, case):
    width = 128 if case == "cloud_in_second_window" else 64
    clean = np.random.default_rng(41).integers(50, 180, (4, 64, width), dtype=np.uint8)
    clouds = np.zeros((64, width), np.uint8)
    if case == "small_scrap":
        clouds[20, 20] = 1  # below the 5-pixel minimum at 60 m resolution
    elif case == "cloud_in_second_window":
        clouds[20:30, 84:94] = 1
    transform = from_origin(1000, 10000, 60, 60)
    for folder in ("candidates", "masks", "pan", "cldDatabase"):
        (tmp_path / folder).mkdir()

    def write_raster(path, data):
        if data.ndim == 2:
            data = data[None]
        with rasterio.open(tmp_path / path, "w", driver="GTiff", height=64, width=width,
                           count=data.shape[0], dtype=data.dtype, crs="EPSG:3826",
                           transform=transform) as dst:
            dst.write(data)

    write_raster("mosaic_xs.tif", clean)
    write_raster("mosaic_pan.tif", clean[0])
    write_raster("mosaic_cloudmask.tif", clouds)
    write_raster("land.tif", np.ones((64, width), np.uint8))
    write_raster("candidates/A0000001.img", clean)
    write_raster("masks/A0000001.tif", np.ones((64, width), np.uint8))
    write_raster("pan/A0000002.tif", clean[0])
    window_count = width // 64
    windows = np.empty(window_count, dtype=object)
    for index in range(window_count):
        windows[index] = Window(index * 64, 0, 64, 64)
    np.save(tmp_path / "allWindow.npy", windows)
    origins = [[1000, 10000], [1000, 10000]]
    if window_count == 2:
        origins.append([1000 + 64 * 60, 10000])
    np.save(tmp_path / "transform.npy", origins)
    (tmp_path / "ProcessOrder.txt").write_text(
        "primary_a.img,primary_b.img" + (",primary_c.img" if window_count == 2 else "")
    )
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

    step4.main()

    with rasterio.open(tmp_path / "mosaic_cloudmask.tif") as result:
        actual = result.read(1)
    if case == "cloud_in_second_window":
        assert np.all(actual[20:30, 84:94] == 0)
        assert np.all(actual[:, :64] == 0)
    else:
        np.testing.assert_array_equal(actual, clouds)
        with rasterio.open(tmp_path / "mosaic_xs.tif") as result:
            np.testing.assert_array_equal(result.read(), clean)
    assert list((tmp_path / "cldDatabase").iterdir()) == []
