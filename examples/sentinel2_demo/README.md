# Sentinel-2 demo (Steps 2–4)

The SPOT and Formosat imagery used in the paper cannot be redistributed
(contact the Center for Space and Remote Sensing Research, National Central
University, for access). This demo runs the same pipeline on free Sentinel-2
imagery, so that anyone can check that it works.

```bash
pip install -r requirements.txt   # from the repository root
python examples/sentinel2_demo/run_demo.py --workdir demo_output
```

The script needs an internet connection but no account. It takes a few minutes
on a laptop (Windows, macOS or Linux) and about 150 MB of disk space.

## What it does

1. **Downloads** small subsets of six Sentinel-2 L2A acquisitions over
   central-western Taiwan (Taichung, about 22 km × 22 km) from the public
   [Earth Search](https://earth-search.aws.element84.com/v1) catalogue and
   reprojects them to TWD97 (`EPSG:3826`), the pipeline's default grid:
   - three partly cloudy scenes (2023-09-22, 2023-10-22, 2023-11-26), each cut
     to a tilted north–south strip that imitates a separate satellite swath;
   - three mostly clear scenes (2023-10-12, 2023-11-01, 2023-11-21) used as
     cloud-filling candidates.
2. **Converts** them to the pipeline's inputs:
   - XS scenes (`a0000001.img`, …): blue, green, red and NIR at 20 m, scaled to
     12-bit;
   - Pan scenes (`raw_pan/A0000002.tif`, …): Sentinel-2 has no panchromatic
     band, so the mean of the 10 m visible bands stands in for it;
   - cloud masks (0 = clear, 255 = cloud) from the Sentinel-2 scene
     classification, buffered by 200 m to cover haze at cloud edges;
   - an all-land land/sea mask.
3. **Runs** Step 2 (seamline mosaic), Step 3 (Pan mosaic) and Step 4 (cloud
   filling) in `demo_output/step2`, `step3` and `step4`, copying files between
   steps as described in the main README.

Step 1 (ERDAS IMAGINE color balancing) is skipped; the scenes go into Step 2
as delivered.

## Outputs

In `demo_output/step4/`:

| File | Content |
|------|---------|
| `mosaic_xs_raw.tif`, `mosaic_pan_raw.tif` | Mosaics after Steps 2 and 3, before cloud filling |
| `mosaic_cloudmask_raw.tif` | Cloud mask of the mosaic before filling (1 = cloud) |
| `mosaic_xs.tif`, `mosaic_pan.tif` | Final cloud-filled mosaics |
| `mosaic_cloudmask.tif` | Final cloud mask: 0 = clear or filled, 1 = cloud left unfilled, 10/20 = filled from a candidate that was itself partly cloudy |

With the default settings, Step 4 fills about 16 of 25 cloud patches and
roughly three quarters of the cloudy area. Patches touching the edge of the
mosaic, and patches with no candidate above the SSIM threshold (0.75), are left
unfilled by design.

Run `python run_demo.py --skip-download --workdir demo_output` to rerun the
pipeline on data downloaded earlier.
