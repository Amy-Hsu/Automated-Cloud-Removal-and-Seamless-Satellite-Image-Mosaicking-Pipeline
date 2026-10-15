"""Run Steps 2-4 of the pipeline on freely available Sentinel-2 imagery.

The demo downloads small subsets of six Sentinel-2 L2A acquisitions over
central-western Taiwan from the public Earth Search catalogue (no account
needed), converts them to the inputs the pipeline expects, and then runs the
seamline mosaic (Step 2), Pan mosaic (Step 3) and cloud filling (Step 4).

Step 1 (ERDAS IMAGINE color balancing) is skipped: the scenes go into Step 2
as delivered.

Usage:
    python run_demo.py                 # download data and run everything
    python run_demo.py --workdir DIR   # choose where data and outputs go
    python run_demo.py --skip-download # reuse data downloaded earlier to --workdir
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import urllib.request

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from scipy.ndimage import binary_dilation

REPO = pathlib.Path(__file__).resolve().parents[2]
STAC_ITEM = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/{}"

CRS = "EPSG:3826"            # TWD97 / TM2, the pipeline's default canvas CRS
XS_RES, PAN_RES = 20.0, 10.0
TRUE_BITS = 12
# Whole demo area (x_min, y_min, x_max, y_max) in EPSG:3826, about 22 km x 22 km
AOI = (200000, 2661000, 222000, 2683000)

# Partly cloudy scenes to mosaic. Each is cut to a tilted north-south strip
# (centre x, centre y, width, height in metres) that imitates the footprint of
# an orthorectified satellite swath. Neighbouring strips overlap by about 3 km,
# and their boundaries cross at two points, where each seamline starts and ends.
PRIMARY = [
    ("a0000001", "S2B_51QTG_20231126_0_L2A", (205500, 2672000, 8000, 18000)),
    ("a0000003", "S2A_51QTG_20231022_0_L2A", (210500, 2672500, 8000, 16000)),
    ("a0000005", "S2A_51QTG_20230922_0_L2A", (215500, 2671500, 8000, 17000)),
]
TILT_DEG = 8
# Mostly clear scenes covering the whole area, used to fill clouds in Step 4.
CANDIDATES = [
    ("a0000101", "S2A_51QTG_20231121_0_L2A", AOI),
    ("a0000103", "S2A_51QTG_20231101_0_L2A", AOI),
    ("a0000105", "S2A_51QTG_20231012_0_L2A", AOI),
]
# Sentinel-2 scene classification classes treated as cloud:
# cloud shadow, cloud medium/high probability, thin cirrus
SCL_CLOUD = [3, 8, 9, 10]
# The classification misses the haze around cloud edges, so masks are buffered
# by this many XS pixels (200 m); Step 4's Poisson blending then takes its
# boundary from clear ground.
CLOUD_BUFFER_PX = 10


def pan_name(xs_name):
    """Pan scenes are numbered one after their XS scene, as for SPOT."""
    return "A" + str(int(xs_name[1:]) + 1).zfill(7)


def footprint(cx, cy, width, height):
    """Corners of a rectangle rotated by TILT_DEG, and its bounds snapped to 100 m."""
    a = np.deg2rad(TILT_DEG)
    corners = [
        (cx + dx * np.cos(a) - dy * np.sin(a), cy + dx * np.sin(a) + dy * np.cos(a))
        for dx, dy in ((-width / 2, -height / 2), (width / 2, -height / 2),
                       (width / 2, height / 2), (-width / 2, height / 2))
    ]
    xs, ys = zip(*corners)
    bounds = (np.floor(min(xs) / 100) * 100, np.floor(min(ys) / 100) * 100,
              np.ceil(max(xs) / 100) * 100, np.ceil(max(ys) / 100) * 100)
    return corners, bounds


def grid(bounds, res):
    x0, y0, x1, y1 = bounds
    return from_origin(x0, y1, res, res), int(round((x1 - x0) / res)), int(round((y1 - y0) / res))


def read_asset(href, bounds, res, resampling):
    transform, width, height = grid(bounds, res)
    with rasterio.open(href) as src, WarpedVRT(
        src, crs=CRS, transform=transform, width=width, height=height, resampling=resampling
    ) as vrt:
        return vrt.read(1)


def write(path, array, bounds, res, driver):
    transform, width, height = grid(bounds, res)
    array = array if array.ndim == 3 else array[None]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver=driver, width=width, height=height, count=array.shape[0],
        dtype=array.dtype, crs=CRS, transform=transform,
    ) as dst:
        dst.write(array)


def to_12bit(raw, offset):
    """L2A digital numbers (reflectance x 10000 + offset) -> 12-bit DN, 0 = no data."""
    dn = np.clip(raw.astype(np.int32) - offset, 1, 2 ** TRUE_BITS - 1).astype(np.uint16)
    dn[raw == 0] = 0
    return dn


def inside(corners, bounds, res):
    """Boolean raster that is True inside the polygon *corners*."""
    transform, width, height = grid(bounds, res)
    polygon = {"type": "Polygon", "coordinates": [corners + [corners[0]]]}
    return geometry_mask([polygon], (height, width), transform, invert=True)


def prepare_scene(name, item_id, rect, xs_dir, mask_dir, pan_dir):
    print(f"  {name} <- {item_id}")
    if rect == AOI:
        corners, bounds = None, AOI
    else:
        corners, bounds = footprint(*rect)
    item = json.load(urllib.request.urlopen(STAC_ITEM.format(item_id)))
    assets = item["assets"]
    # Processing baseline 04.00+ adds 1000 to every DN; Earth Search may already have removed it.
    props = item["properties"]
    newer = float(props.get("s2:processing_baseline", "0")) >= 4
    offset = 1000 if newer and not props.get("earthsearch:boa_offset_applied", False) else 0
    # XS band order follows SPOT 6/7: blue, green, red, near-infrared
    xs = np.stack([
        to_12bit(read_asset(assets[b]["href"], bounds, XS_RES, Resampling.average), offset)
        for b in ("blue", "green", "red", "nir")
    ])
    valid = np.all(xs > 0, axis=0)
    if corners:
        valid &= inside(corners, bounds, XS_RES)
    xs[:, ~valid] = 0
    write(xs_dir / f"{name}.img", xs, bounds, XS_RES, "HFA")

    scl = read_asset(assets["scl"]["href"], bounds, XS_RES, Resampling.nearest)
    cloud = binary_dilation(np.isin(scl, SCL_CLOUD), iterations=CLOUD_BUFFER_PX)
    cloud = np.where(cloud & valid, 255, 0).astype(np.uint8)
    write(mask_dir / f"{name}_cloud.tif", cloud, bounds, XS_RES, "GTiff")

    # Sentinel-2 has no panchromatic band: use the mean of the 10 m visible bands.
    vis = [read_asset(assets[b]["href"], bounds, PAN_RES, Resampling.nearest) for b in ("blue", "green", "red")]
    pan = np.mean([to_12bit(v, offset) for v in vis], axis=0).astype(np.uint16)
    pan[np.any([v == 0 for v in vis], axis=0)] = 0
    if corners:
        pan[~inside(corners, bounds, PAN_RES)] = 0
    write(pan_dir / f"{pan_name(name)}.tif", pan, bounds, PAN_RES, "GTiff")
    return float((cloud > 0).mean())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=1))


def download(data):
    """Download and convert all scenes into *data* (only needed once)."""
    print("Downloading Sentinel-2 scenes to mosaic...")
    for name, item, rect in PRIMARY:
        cloud = prepare_scene(name, item, rect, data / "scenes", data / "cloud_masks", data / "raw_pan")
        print(f"    cloud cover {cloud:.0%}")
    print("Downloading cloud-filling candidates...")
    for name, item, rect in CANDIDATES:
        prepare_scene(name, item, rect, data / "candidates", data / "cloud_masks", data / "raw_pan")
    # The demo area is all land, so the land/sea mask is all ones.
    _, w, h = grid(AOI, XS_RES)
    write(data / "land_sea_mask.tif", np.ones((h, w), np.uint8), AOI, XS_RES, "GTiff")


def set_up_steps(work, data):
    """Create fresh step folders, laid out as described in the main README."""
    s2, s3, s4 = work / "step2", work / "step3", work / "step4"
    for d in (s2, s3, s4):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    (s2 / "NSPO_GRID_vectorization").mkdir()
    (s3 / "new_pan").mkdir()
    (s4 / "cldDatabase").mkdir()
    for name, _, _ in PRIMARY:
        shutil.copy(data / "scenes" / f"{name}.img", s2)
    shutil.copytree(data / "cloud_masks", s2 / "cloud_masks")
    shutil.copytree(data / "raw_pan", s3 / "raw_pan")
    # Step 4 candidates: only the clear scenes. (The Sentinel-2 cloud masks of
    # the mosaicked scenes miss some haze, which would then be copied back in.)
    shutil.copytree(data / "candidates", s4 / "cloud_database")
    shutil.copytree(data / "cloud_masks", s4 / "cloud_masks")
    shutil.copytree(data / "raw_pan", s4 / "raw_pan")
    shutil.copy(data / "land_sea_mask.tif", s4)


def write_configs(work):
    s2, s3, s4 = work / "step2", work / "step3", work / "step4"
    canvas = {"canvas_crs": CRS, "canvas_x_max": str(AOI[2]), "canvas_y_min": str(AOI[1])}
    write_json(s2 / "para.txt", {
        "path_method": "a6c", "res": str(XS_RES), "pan_res": str(PAN_RES), "resampleTo": "5",
        "trueBits": str(TRUE_BITS), "imgPath": str(s2), "kernel": "11", **canvas,
    })
    write_json(s3 / "para_pan.txt", {
        "res": str(PAN_RES), "trueBits": str(TRUE_BITS), "StartLetter": "a",
        "upscale_factor": str(int(XS_RES / PAN_RES)), **canvas,
    })
    write_json(s4 / "para_cld.txt", {
        "res": str(XS_RES), "trueBits": str(TRUE_BITS),
        "CandiImgPath": str(s4 / "cloud_database"), "CldMaskPath": str(s4 / "cldDatabase"),
        "MosaicImgAddress": str(s4 / "mosaic_xs.tif"), "MosaicCldAddress": str(s4 / "mosaic_cloudmask.tif"),
        "CandiMaskFolderName": "cloud_masks", "PanMosaicImgAddress": str(s4 / "mosaic_pan.tif"),
        "pan_res": str(PAN_RES), "seaBorderAddress": str(s4 / "land_sea_mask.tif"),
        "PanImgAddress": str(s4 / "raw_pan"), "cloud_tolerance": "0.25", "ssim_threshold": "0.75",
        "canvas_crs": CRS,
    })


def run_step(script, cwd):
    print(f"\n=== Running {script} in {cwd} ===")
    subprocess.run([sys.executable, str(REPO / script)], cwd=cwd, check=True)


def copy_matching(src, dst, patterns):
    for pattern in patterns:
        for f in src.glob(pattern):
            shutil.copy(f, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", default="demo_output", type=pathlib.Path)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    work = args.workdir.resolve()
    s2, s3, s4 = work / "step2", work / "step3", work / "step4"

    data = work / "data"
    if not args.skip_download:
        download(data)
    set_up_steps(work, data)
    write_configs(work)

    run_step("step2_mosaic/step2mosaic.py", s2)

    copy_matching(s2, s3, ["transform.npy", "ImgInd.npy", "allWindow.npy", "Union_gt.npy",
                           "ProcessOrder.txt", "*_seamline_polygon.img", "*_seamline_polygon.img.aux.xml"])
    run_step("step3_pan_mosaic/mosaicPan_block_rs.py", s3)

    copy_matching(s2, s4, ["mosaic_xs.tif", "mosaic_cloudmask.tif", "allWindow.npy",
                           "transform.npy", "ProcessOrder.txt"])
    copy_matching(s3, s4, ["mosaic_pan.tif"])
    for f in ("mosaic_xs.tif", "mosaic_pan.tif", "mosaic_cloudmask.tif"):
        shutil.copy(s4 / f, s4 / f.replace(".tif", "_raw.tif"))
    run_step("step4_cloud_removal/step4cloud_removal.py", s4)

    print(f"\nDone. Final products are in {s4}:")
    for f in ("mosaic_xs.tif", "mosaic_pan.tif", "mosaic_cloudmask.tif"):
        print("  ", s4 / f)


if __name__ == "__main__":
    main()
