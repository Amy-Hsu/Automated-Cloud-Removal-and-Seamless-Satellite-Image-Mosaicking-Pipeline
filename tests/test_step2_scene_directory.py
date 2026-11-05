"""Step 2 can read scenes outside its working directory."""
import ast
import json

import numpy as np
import rasterio
from rasterio.shutil import copy
from rasterio.transform import from_origin


def test_step2_reads_separate_scene_directory(step2, tmp_path, monkeypatch):
    names = ["a0000003.img", "a0000001.img", "a0000005.img"]
    monkeypatch.setattr(step2, "listdir", lambda _path: names.copy())
    # Exercise raster I/O and polygonization without requiring the GDAL CLI.
    monkeypatch.setattr(step2, "assign_crs_copy", lambda src, dst, crs: copy(src, dst, driver="GTiff"))

    def run(work, *, separate):
        work.mkdir()
        inputs = work / "separate scenes" if separate else work
        inputs.mkdir(exist_ok=True)
        masks = work / step2.CLOUD_MASK_FOLDER
        masks.mkdir()
        (work / "NSPO_GRID_vectorization").mkdir()
        res, left, top, size = 2, 200000, 2500000, 160
        crs = "EPSG:3826"

        def write(path, data, transform, driver):
            with rasterio.open(
                path, "w", driver=driver, height=size, width=size,
                count=data.shape[0], dtype=data.dtype, crs=crs, transform=transform,
            ) as dst:
                dst.write(data)

        for index, y_offset in enumerate((0, 60, 0)):
            stem = f"a000000{1 + index * 2}"
            transform = from_origin(left + index * 100, top - y_offset, res, res)
            pixels = np.random.default_rng(index).integers(70, 130, (4, size, size), dtype=np.uint8)
            write(inputs / f"{stem}.img", pixels, transform, "HFA")
            write(masks / f"{stem}.tif", np.zeros((1, size, size), dtype=np.uint8), transform, "GTiff")

        (work / "para.txt").write_text(json.dumps({
            "path_method": "a6c_int", "resampleTo": 2, "res": res, "trueBits": 8,
            "imgPath": inputs.name if separate else ".", "kernel": 5, "pan_res": 1,
            "canvas_crs": crs, "canvas_x_max": left + size * res + 200,
            "canvas_y_min": top - size * res - 120,
        }), encoding="utf-8")
        with monkeypatch.context() as mp:
            mp.chdir(work)
            step2.main()
        assert ast.literal_eval((work / "ProcessOrder.txt").read_text()) == names
        assert (work / "a0000003a0000001_seamline_polygon.img").exists()
        assert (work / "a0000005_seamline_polygon.img").exists()
        with rasterio.open(work / "mosaic_xs.tif") as mosaic:
            pixels = mosaic.read()
        with rasterio.open(work / "mosaic_cloudmask.tif") as mask:
            clouds = mask.read()
        return pixels, clouds, np.load(work / "transform.npy")

    expected = run(tmp_path / "control", separate=False)
    actual = run(tmp_path / "external", separate=True)
    assert np.any(expected[0])
    for result, control in zip(actual, expected):
        np.testing.assert_array_equal(result, control)
