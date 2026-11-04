"""Sensor bit depths need a real unsigned storage dtype in every pipeline step."""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.mark.parametrize("step_name", ["step2", "step3", "step4"])
@pytest.mark.parametrize("bits,expected", [
    (1, "uint8"), (7, "uint8"), (8, "uint8"),
    (9, "uint16"), (10, "uint16"), (11, "uint16"), (12, "uint16"),
    (14, "uint16"), (15, "uint16"), (16, "uint16"),
    (17, "uint32"), (24, "uint32"), (32, "uint32"),
    (33, "uint64"), (64, "uint64"),
])
def test_sensor_values_fit_the_selected_raster_dtype(request, tmp_path, step_name, bits, expected):
    step = request.getfixturevalue(step_name)
    dtype = np.dtype(step.image_dtype(bits))
    assert dtype.name == expected
    pixels = np.array([[0, (1 << bits) - 1]], dtype=dtype)
    path = tmp_path / "sensor.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=1, width=2, count=1,
        dtype=dtype, crs="EPSG:3826", transform=from_origin(1000, 1000, 1, 1),
    ) as dst:
        dst.write(pixels, 1)
    with rasterio.open(path) as src:
        assert src.dtypes == (expected,)
        np.testing.assert_array_equal(src.read(1), pixels)


@pytest.mark.parametrize("step_name", ["step2", "step3", "step4"])
@pytest.mark.parametrize("bits", [-1, 0, 65])
def test_unsupported_sensor_bit_depth_has_a_clear_error(request, step_name, bits):
    step = request.getfixturevalue(step_name)
    with pytest.raises(ValueError, match="trueBits"):
        step.image_dtype(bits)
