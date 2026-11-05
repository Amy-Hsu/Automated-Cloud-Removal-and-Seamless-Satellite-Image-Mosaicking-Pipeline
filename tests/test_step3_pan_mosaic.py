"""Exercise the Step 3 mosaic with synthetic, already-clipped Pan scenes."""

import json

import numpy as np
import rasterio
from rasterio.transform import from_origin


def test_main_blends_initial_pair_and_updates_overlapping_window(step3, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    crs = "EPSG:3826"
    left, top = 200000, 2500000
    res = 2
    initial_transform = from_origin(left, top, res, res)
    update_transform = from_origin(left + 2 * res, top, res, res)
    (tmp_path / "para_pan.txt").write_text(json.dumps({
        "res": res,
        "trueBits": 12,
        "StartLetter": "a",
        "upscale_factor": 4,
        "canvas_crs": crs,
        "canvas_x_max": left + 6 * res,
        "canvas_y_min": top - 4 * res,
    }), encoding="utf-8")
    (tmp_path / "ProcessOrder.txt").write_text(
        "a0000001.img,a0000003.img,a0000005.img", encoding="utf-8",
    )
    np.save("transform.npy", [[left, top], [left, top], [left + 2 * res, top]])
    np.save("ImgInd.npy", [
        [(1, "a0000003.img", 8, "b1", "b1d")],
        [(1, "a0000005.img", 8, "b1d", "b1")],
    ])
    # Step 3 loads these Step 2 metadata files even though it does not use them.
    np.save("allWindow.npy", [])
    np.save("Union_gt.npy", [])

    def write_raster(path, data, transform, driver, **options):
        with rasterio.open(
            path, "w", driver=driver, count=1, height=4, width=4,
            dtype=data.dtype, crs=crs, transform=transform, **options,
        ) as dst:
            dst.write(data, 1)

    (tmp_path / "new_pan").mkdir()
    for name, value, transform in (
        ("a0000002", 1000, initial_transform),
        ("a0000004", 2000, initial_transform),
        ("a0000006", 3000, update_transform),
    ):
        write_raster(
            f"new_pan/{name}.img.tif", np.full((4, 4), value, dtype=np.uint16),
            transform, "GTiff", blockysize=1,
        )

    seamline = np.tile(np.array([0, 0.5, 1, 1], dtype=np.float32), (4, 1))
    write_raster("a0000001a0000003_seamline_polygon.img", seamline, initial_transform, "HFA")
    write_raster("a0000005_seamline_polygon.img", seamline, update_transform, "HFA")

    step3.main()

    with rasterio.open("mosaic_pan.tif") as mosaic:
        assert mosaic.dtypes == ("uint16",)
        assert mosaic.crs.to_epsg() == 3826
        assert mosaic.transform == initial_transform
        expected = np.tile([1000, 1500, 2000, 2500, 3000, 3000], (4, 1))
        np.testing.assert_array_equal(mosaic.read(1), expected)
