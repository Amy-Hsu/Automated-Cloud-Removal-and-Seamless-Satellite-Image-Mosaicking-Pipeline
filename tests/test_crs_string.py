"""Steps 2-4 pass canvas_crs to rasterio as a string, not the deprecated {'init': ...} dict."""
import re

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from conftest import ROOT

SCRIPTS = [
    "step2_mosaic/step2mosaic.py",
    "step3_pan_mosaic/mosaicPan_block_rs.py",
    "step4_cloud_removal/step4cloud_removal.py",
]


@pytest.mark.parametrize("relpath", SCRIPTS)
def test_no_init_crs_dict(relpath):
    source = (ROOT / relpath).read_text()
    assert not re.search(r"\{\s*['\"]init['\"]\s*:", source)


@pytest.mark.parametrize("canvas_crs", ["EPSG:3826", "EPSG:32617", "EPSG:4326"])
def test_crs_string_round_trips(tmp_path, canvas_crs):
    path = tmp_path / "canvas.tif"
    with rasterio.open(path, "w", driver="GTiff", height=2, width=2, count=1,
                       dtype="uint8", crs=canvas_crs,
                       transform=from_origin(0, 2, 1, 1)) as dst:
        dst.write(np.zeros((1, 2, 2), dtype=np.uint8))
    with rasterio.open(path) as src:
        assert src.crs.to_string() == canvas_crs
