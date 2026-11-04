"""Steps 2 and 3 run GDAL tools with argument lists, so paths may contain spaces (#6)."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize("step_name", ["step2", "step3"])
def test_run_tool_passes_paths_with_spaces_as_single_arguments(request, tmp_path, step_name):
    step = request.getfixturevalue(step_name)
    path = tmp_path / "folder with spaces" / "scene 1.tif"
    out = tmp_path / "argv.txt"
    script = "import sys, pathlib; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])"
    step.run_tool([sys.executable, "-c", script, out, path])
    assert out.read_text() == str(path)


@pytest.mark.parametrize("step_name", ["step2", "step3"])
def test_run_tool_raises_when_the_tool_fails(request, step_name):
    step = request.getfixturevalue(step_name)
    with pytest.raises(subprocess.CalledProcessError):
        step.run_tool([sys.executable, "-c", "raise SystemExit(3)"])


def test_assign_crs_copy_builds_gdal_translate_arguments(step2, monkeypatch):
    calls = []
    monkeypatch.setattr(step2, "run_tool", calls.append)
    step2.assign_crs_copy("in dir/a0000001.img", "out dir/a0000001.tif", "EPSG:3826")
    assert calls == [["gdal_translate", "-a_srs", "EPSG:3826", "-of", "GTiff",
                      "in dir/a0000001.img", "out dir/a0000001.tif"]]


def test_clip_pan_builds_gdalwarp_arguments(step3, monkeypatch):
    calls = []
    monkeypatch.setattr(step3, "run_tool", calls.append)
    step3.clip_pan("raw pan/A0000002.tif", "new pan/A0000002.img.tif", "EPSG:3826",
                   (1.0, 2.0, 3.0, 4.0))
    assert calls == [["gdalwarp", "-t_srs", "EPSG:3826", "-te", 1.0, 2.0, 3.0, 4.0,
                      "raw pan/A0000002.tif", "new pan/A0000002.img.tif"]]
