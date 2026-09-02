# Automated Cloud Removal and Seamless Satellite Image Mosaicking Pipeline

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.acags.2026.100387-blue)](https://doi.org/10.1016/j.acags.2026.100387)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, largely automated pipeline that turns individual orthorectified satellite scenes into a single color-balanced, seamless, cloud-filled mosaic — for both the multispectral (XS) and panchromatic (Pan) products, together with a mosaic-wide cloud mask.

If you use this code, please cite the accompanying paper:

> Hsu, H.-J., Tseng, K.-H., Tsai, F., Liu, C.-L., Lo, C.-C., & Moortgat, J. (2026). A novel approach to automated cloud removal and seamless multisensor satellite image mosaicking. *Applied Computing and Geosciences*, 100387. https://doi.org/10.1016/j.acags.2026.100387

**Golden rule for every step: no spaces anywhere in folder or file paths.**

---

## Pipeline overview

| Step | Folder | What it does | Main tooling |
|------|--------|--------------|--------------|
| 1 | `step1_color_balance/` | Radiometric normalization (color balance) of each raw scene against reference imagery, with dark-area detail preservation | ERDAS IMAGINE 2014 spatial models + `MappingReshapeXS.exe` |
| 2 | `step2_mosaic/` | Automatic seamline detection (sparse-node Dijkstra over image-difference cost) and seamless mosaicking of the color-balanced XS scenes; builds the mosaic-wide cloud mask | Python (`step2mosaic.py`) + `GetBoundaryMask.exe` |
| 3 | `step3_pan_mosaic/` | Mosaics the corresponding Pan scenes reusing the seamlines found in Step 2 | Python (`mosaicPan_block_rs.py`) |
| 4 | `step4_cloud_removal/` | Automatically fills cloud-contaminated areas of the XS and Pan mosaics from a database of candidate scenes, using SSIM-based similarity screening and Poisson-style blending | Python (`step4cloud_removal.py`) |

Outputs after Step 4:

- `mosaic_xs.tif` — final cloud-filled multispectral mosaic (4 bands)
- `mosaic_pan.tif` — final cloud-filled panchromatic mosaic
- `mosaic_cloudmask.tif` — final mosaic cloud mask (filled clouds = `(loop_index+1)*10`; unprocessed clouds remain `1`)
- `*_raw.tif` copies of the pre-Step-4 mosaics for comparison

## Installation

```bash
# Create a conda environment (recommended)
conda create -n mosaic python=3.9
conda activate mosaic

# Install geospatial libraries via conda-forge
conda install -c conda-forge gdal rasterio geopandas shapely

# Install remaining packages
pip install -r requirements.txt
```

## Requirements

- **Windows 10** (the pipeline shells out to `.bat`/`.exe` and uses Windows path separators)
- **ERDAS IMAGINE 2014** for Step 1 (later versions, e.g. 2018/2020, may produce wrong colors)
- **MATLAB Runtime R2018a (v9.4)** — required by `GetBoundaryMask.exe` in Step 2
- **GDAL command-line tools** on `PATH` (`gdalwarp`, `gdal_translate`, `gdal_polygonize.py`, `gdal_edit.py`)
- **Python 3.9** with the packages in `requirements.txt`

Input data (not included): orthorectified satellite XS scenes (`.raw`/`.img`) with matching Pan scenes and per-scene cloud masks (`.ers` or `.tif`), plus the Step-1 reference images.

> **Configurable canvas extents:** the mosaic canvas bounds are set in each step's configuration file (`canvas_x_max`, `canvas_y_min`, `canvas_crs`). The defaults use the Taiwan TWD97 coordinate system (`EPSG:3826`). To mosaic imagery in another region, update these parameters in `para.txt`, `para_pan.txt`, and `para_cld.txt`.

> **Land/sea mask:** Step 4 requires a binary land/sea border mask raster that matches the XS mosaic grid. This file is **not included**. Create or obtain one for your study area and set its path via `seaBorderAddress` in `para_cld.txt`.

---

## Step 1 — Color balance (ERDAS IMAGINE 2014)

Folder: `step1_color_balance/`

> **Note:** the ERDAS spatial models (`.gmdx`), batch configuration files (`.bcf`), and the compiled boundary-mask tools (`MappingReshapeXS.exe`, `GetBoundaryMask.exe`, `BndPolygonize_v.bat`) were originally developed at the **Center for Space and Remote Sensing Research (CSRSR), National Central University** and the **National Space Organization (NSPO/TASA), Taiwan**. Python source for the table-remapping step is provided in `distrib/runrunrun.py`.

Workflow (drive it with `Auto_2data_test.bat`):

1. **Step1SPOTf** (spatial model `color_balance.gmdx` / snapshot `SM_BATCH_GMDX_012516`): per scene, finds invariant areas against three reference layers and produces a mask, four color lookup tables `col1..4_N.tbl`, and a first balanced image.
2. **`distrib/runrunrun.exe`** (source: `distrib/runrunrun.py`): calls `MappingReshapeXS.exe` on each table group to stretch near-zero DN values so dark areas keep detail → `colX_N.fd.tbl`.
3. **Step2SPOTf** (spatial model `applytable_taibor_noedge.gmdx` / snapshot `SM_BATCH_GMDX_023448`): applies the remapped tables → the final color-balanced scene.

Setup on a new machine:

1. Edit the output folders inside `Step1SPOTf.bcf` / `Step2SPOTf.bcf` and make the `smprocess` line point at the `SM_BATCH_GMDX_*` files (absolute path, no spaces).
2. Build your scene lists from `Step1SPOTf.bls.example` / `Step2SPOTf.bls.example` (tab-separated, one row per scene) and save them as `.bls`.
3. In ERDAS IMAGINE → *Batch*, load the `.bcf`+`.bls` pairs and submit once — this generates `Step1SPOTf.bat` / `Step2SPOTf.bat` for **your** machine (these are machine-specific and intentionally not shipped).
4. From then on, run everything with `Auto_2data_test.bat`.

## Step 2 — Seamless multispectral mosaic

Folder: `step2_mosaic/`. Working-folder layout:

```
step2/
├── run.bat
├── step2mosaic.py
├── para.txt
├── a0000001.img  a0000002.img  ...   # color-balanced scenes from Step 1
├── NSPO_GRID_vectorization/          # GetBoundaryMask.exe + BndPolygonize_v.bat (do not delete)
└── cloud_masks/                      # per-scene cloud masks (.tif preferred over .ers)
```

`para.txt` parameters:

| Key | Meaning |
|-----|---------|
| `path_method` | Seamline cost function: `a6`, `a7`, `a6c`, `a7c` (`a6c` is a good default) |
| `res` | XS resolution (m), e.g. `6` |
| `pan_res` | Pan resolution (m), e.g. `1.5` |
| `resampleTo` | Downsampling factor for seamline detection (recommended `5`) |
| `trueBits` | Actual bit depth of the DN values, e.g. `12` |
| `imgPath` | Absolute path of the folder containing the scenes |
| `kernel` | Shrink kernel for seamline detection (odd number, recommended `61`) |
| `canvas_crs` | CRS of the mosaic canvas (e.g. `EPSG:3826`) |
| `canvas_x_max` | Eastern bound of the mosaic canvas (map units) |
| `canvas_y_min` | Southern bound of the mosaic canvas (map units) |

Outputs: `mosaic_xs.tif` (XS mosaic), `mosaic_cloudmask.tif` (cloud mask mosaic), `ProcessOrder.txt`, `transform.npy`, `allWindow.npy`, `ImgInd.npy`, `Union_gt.npy`, per-pair seamline and blend-polygon rasters.

## Step 3 — Panchromatic mosaic

Folder: `step3_pan_mosaic/`.

Copy from Step 2: `transform.npy`, `ImgInd.npy`, `allWindow.npy`, `Union_gt.npy`, `ProcessOrder.txt`, and all `*_seamline_polygon.img` (+`.ige`) files. Add `raw_pan/` and an empty `new_pan/`.

Configure `para_pan.txt` (`res`, `trueBits`, `StartLetter`, `upscale_factor`, `canvas_crs`, `canvas_x_max`, `canvas_y_min`) and run `run.bat`. Output: `mosaic_pan.tif`.

## Step 4 — Automatic cloud filling

Folder: `step4_cloud_removal/`.

```
step4/
├── run.bat
├── step4cloud_removal.py
├── para_cld.txt
├── mosaic_xs.tif  mosaic_cloudmask.tif   # from Step 2
├── mosaic_pan.tif                         # from Step 3
├── allWindow.npy  transform.npy  ProcessOrder.txt   # from Step 2
├── land_sea_mask.img                      # user-provided; must match XS grid
├── cloud_database/                        # candidate XS scenes for filling
├── cloud_masks/                           # cloud-mask database for the candidates
├── raw_pan/                               # candidate Pan scenes
└── cldDatabase/                           # empty working folder
```

Run with `run.bat`. See `docs/TROUBLESHOOTING.md` for known failure modes and their fixes.

## Troubleshooting

See [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

## Repository layout

```
step1_color_balance/   ERDAS models, batch configs, table-remapping tool (+ runrunrun.py source)
step2_mosaic/          seamline + XS mosaic script, boundary-mask tools, para.txt template
step3_pan_mosaic/      Pan mosaic script, para_pan.txt template
step4_cloud_removal/   cloud-filling script, para_cld.txt template
docs/                  troubleshooting notes
paper.md               JOSS paper source
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Released under the MIT License (see [`LICENSE`](LICENSE)).
