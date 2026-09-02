# Automated Cloud Removal and Seamless Satellite Image Mosaicking Pipeline

An end-to-end, largely automated pipeline that turns individual orthorectified SPOT-6/7 scenes into a single color-balanced, seamless, cloud-filled mosaic of Taiwan — for both the multispectral (XS, 6 m) and panchromatic (Pan, 1.5 m) products, together with a mosaic-wide cloud mask.

If you use this code, please cite the accompanying paper:

> Hsu, H.-J., Tseng, K.-H., Tsai, F., Liu, C.-L., Lo, C.-C., & Moortgat, J. (2026). A novel approach to automated cloud removal and seamless multisensor satellite image mosaicking. *Applied Computing and Geosciences*, 100387. https://doi.org/10.1016/j.acags.2026.100387

⚠️ **Golden rule for every step: no spaces anywhere in folder or file paths.**

---

## Pipeline overview

| Step | Folder | What it does | Main tooling |
|------|--------|--------------|--------------|
| 1 | `step1_color_balance/` | Radiometric normalization (color balance) of each raw scene against reference imagery, with dark-area detail preservation | ERDAS IMAGINE 2014 spatial models + `MappingReshapeXS.exe` |
| 2 | `step2_mosaic/` | Automatic seamline detection (graph A\* over image-difference cost) and seamless mosaicking of the color-balanced XS scenes; builds the mosaic-wide cloud mask | Python (`step2mosaic.py`) + `GetBoundaryMask.exe` |
| 3 | `step3_pan_mosaic/` | Mosaics the corresponding Pan scenes re-using the seamlines found in Step 2 | Python (`mosaicPan_block_rs.py`) |
| 4 | `step4_cloud_removal/` | Automatically fills cloud-contaminated areas of the XS and Pan mosaics from a database of candidate scenes, using SSIM-based similarity screening and seamless (Poisson-style) blending | Python (`step4cloud_removal.py`) |

Outputs after Step 4:

- `MosTai.tif` — final cloud-filled multispectral mosaic (4 bands)
- `MosTai_Pan.tif` — final cloud-filled panchromatic mosaic
- `TaiMas.tif` — final mosaic cloud mask (filled clouds = `(loop_index+1)*10`; unprocessed clouds remain `1`)
- `*_Raw.tif` copies of the pre-Step-4 mosaics for comparison

## Requirements

- Windows 10 (the pipeline shells out to `.bat`/`.exe` and uses Windows path separators)
- **ERDAS IMAGINE 2014** for Step 1 (later versions, e.g. 2018/2020, may produce wrong colors)
- **MATLAB Runtime R2018a (v9.4)** — required by `GetBoundaryMask.exe` in Step 2
- **GDAL command-line tools** on `PATH` (`gdalwarp`, `gdal_translate`, `gdal_polygonize.py`, `gdal_edit.py`)
- **Python 3.9** with the packages in `requirements.txt` (conda-forge recommended for `gdal`/`rasterio`/`geopandas`)

Input data (not included): orthorectified SPOT-6/7 XS scenes (`aXXXXXXX.raw`/`.img`, 6 m, 12-bit, EPSG:3826) with matching Pan scenes (`AXXXXXXX_B1.DAT.raw`, 1.5 m) and per-scene cloud masks (`.ers` or `.tif`), plus the three Step-1 reference images.

> **Note on `tw_city6m.img`:** Step 4 requires a land/sea border mask raster (`tw_city6m.img`) that matches the XS mosaic grid. This file is **not included** in the repository. You must create or obtain a binary land/sea mask for your study area, resampled to the same resolution and extent as your XS mosaic, and place it in the Step 4 working folder. Set its path in `para_cld.txt` via `seaBorderAddress`.

> **Note on hard-coded extents:** the mosaic canvas is sized with the Taiwan TWD97 bounds hard-coded in the scripts (`x_max = 406000`, `y_min = 2400000`, CRS `EPSG:3826`). To use the pipeline in another region, adapt these constants in `step2mosaic.py`, `mosaicPan_block_rs.py`, and the canvas-related code.

---

## Step 1 — Color balance (ERDAS IMAGINE 2014)

Folder: `step1_color_balance/`

Workflow (drive it with `Auto_2data_test.bat`):

1. **Step1SPOTf** (spatial model `color_balance.gmdx` / snapshot `SM_BATCH_GMDX_012516`): per scene, finds invariant areas against three reference layers and produces a mask, four color lookup tables `col1..4_N.tbl`, and a first balanced image.
   - `Input1` = river reference (e.g. `river6m_new2.img`)
   - `Input2` = Taiwan border reference (e.g. `tai_bor_new2.img`)
   - `Input3` = reference mosaic (e.g. `sp2017xs_12m.img`)
   - `Input4` = the raw scene (`aXXXXXXX.raw`)
2. **`distrib/runrunrun.exe`** (source: `distrib/runrunrun.py`): calls `MappingReshapeXS.exe` on each table group to stretch near-zero DN values so dark areas keep detail → `colX_N.fd.tbl`.
3. **Step2SPOTf** (spatial model `applytable_taibor_noedge.gmdx` / snapshot `SM_BATCH_GMDX_023448`): applies the remapped tables → the final color-balanced scene `aXXXXXXX.img`.

Setup on a new machine:

1. Edit the output folders inside `Step1SPOTf.bcf` / `Step2SPOTf.bcf` and make the `smprocess` line point at the `SM_BATCH_GMDX_*` files (absolute path, no spaces).
2. Build your scene lists from `Step1SPOTf.bls.example` / `Step2SPOTf.bls.example` (tab-separated, one row per scene) and save them as `.bls`.
3. In ERDAS IMAGINE → *Batch*, load the `.bcf`+`.bls` pairs and submit once — this generates `Step1SPOTf.bat` / `Step2SPOTf.bat` for **your** machine (these are machine-specific and intentionally not shipped).
4. From then on, run everything with `Auto_2data_test.bat`.

Tables must be written into the `distrib/` folder so `runrunrun.exe` can find them.

## Step 2 — Seamless multispectral mosaic

Folder: `step2_mosaic/`. Working-folder layout (see `docs/` for the original illustrated guide):

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
| `path_method` | Seamline cost function: `a6`, `a7`, `a6c`, `a7c` (with `resampleTo=5` results are very similar; `a6c` is a good default) |
| `res` | XS resolution (m), e.g. `6` |
| `pan_res` | Pan resolution (m), e.g. `1.5` |
| `resampleTo` | Downsampling factor for seamline detection (recommended `5`) |
| `trueBits` | Actual bit depth of the DN values, e.g. `12` |
| `imgPath` | Absolute path of the folder containing the scenes |
| `kernel` | Shrink kernel for seamline detection (odd number, recommended `61`) |

Run: open CMD in the folder and type `run.bat`.

Scene requirements: image borders must be clean (no stray `1` values outside the footprint) and every consecutive pair must genuinely **intersect** (≥ 2 boundary intersection points) — containment is not supported.

Outputs: `MosTai.tif` (XS mosaic), `TaiMas.tif` (cloud mask mosaic), `ProcessOrder.txt`, `transform.npy`, `allWindow.npy`, `ImgInd.npy`, `Union_gt.npy`, per-pair seamlines `*MosTai_sl.img` and blend polygons `*MosTai_sl_polygon.img`, plus intermediate shapefiles in `NSPO_GRID_vectorization/`.

**Re-running:** delete all generated outputs first, otherwise the run will fail (files inside `NSPO_GRID_vectorization/` may be kept unless that step itself failed).

## Step 3 — Panchromatic mosaic

Folder: `step3_pan_mosaic/`. Working-folder layout:

```
step3/
├── run.bat
├── mosaicPan_block_rs.py
├── para_pan.txt
├── transform.npy  ImgInd.npy  allWindow.npy  Union_gt.npy  ProcessOrder.txt   # copied from Step 2
├── *MosTai_sl_polygon.img(+.ige)      # copied from Step 2 (V scenes → V-1 files)
├── raw_pan/                            # one folder per Pan scene: A000XXXX/A000XXXX_B1.DAT.raw
└── new_pan/                            # empty; receives the re-clipped Pan tiles
```

`para_pan.txt`: `res` (Pan resolution), `trueBits`, `StartLetter` (first letter of the XS scene IDs, e.g. `a`), and `upscale_factor` (XS→Pan resolution ratio, e.g. `4` for 6 m → 1.5 m).

The Pan scene footprints must match the corresponding XS scenes exactly, and `raw_pan` must contain no ERDAS side-car files (`.aux`, `.rrd`, `.rde` — delete them). Run with `run.bat`; output is `MosTai_Pan.tif`.

**Re-running:** delete `MosTai_Pan.tif` first. `new_pan/` can be kept unless the source imagery changed.

## Step 4 — Automatic cloud filling

Folder: `step4_cloud_removal/`. Working-folder layout:

```
step4/
├── run.bat
├── step4cloud_removal.py
├── para_cld.txt
├── MosTai.tif  TaiMas.tif                 # from Step 2
├── MosTai_Pan.tif                          # from Step 3
├── allWindow.npy  transform.npy  ProcessOrder.txt   # from Step 2
├── tw_city6m.img                           # Taiwan land/sea border mask (user-provided; must match XS grid)
├── cloud_database/                         # candidate XS scenes for filling (include the mosaicked ones too)
├── cloud_masks/                            # cloud-mask database for the candidates (.ers preferred over .tif)
├── raw_pan/                                # candidate Pan scenes (include the mosaicked ones too)
└── cldDatabase/                            # empty working folder; clouds >500 px are written here and auto-deleted
```

`para_cld.txt` parameters: `res`, `trueBits`, `pan_res` as before; `CandiImgPath` (candidate XS folder), `CldMaskPath` (working folder), `MosaicImgAddress` / `MosaicCldAddress` / `PanMosaicImgAddress` (the three mosaics), `CandiMaskFolderName` (cloud-mask database, name or absolute path), `seaBorderAddress` (land/sea mask), `PanImgAddress` (candidate Pan folder), `cloud_tolerance(decimal)` (allowed residual cloud fraction in a candidate patch, e.g. `0.25`) and `ssim_threshold(0<val<=1)` (minimum structural similarity between candidate and mosaic, e.g. `0.1`).

Run with `run.bat`. Do not open/modify files inside `cldDatabase/` while it runs. Before running, verify that the color-balanced scenes were not cropped relative to the mosaic grid (a known failure mode — see `docs/TROUBLESHOOTING.md`).

To **iterate** cloud filling: refresh `cloud_database/`, `raw_pan/` and the cloud-mask database with new candidates (used ones can be removed to save time) and re-run. Optionally reset mask values you want reprocessed back to `1` in `TaiMas.tif` (only value `1` gets processed).

## Troubleshooting

See `docs/TROUBLESHOOTING.md` for the known failure modes (bad intersections, "No road from target to endnode", thin overlap regions, cropped color-balanced scenes) and their fixes.

## Repository layout

```
step1_color_balance/   ERDAS models, batch configs, table-remapping tool (+ runrunrun.py source)
step2_mosaic/          seamline + XS mosaic script, boundary-mask tools, para.txt template
step3_pan_mosaic/      Pan mosaic script, para_pan.txt template
step4_cloud_removal/   cloud-filling script, para_cld.txt template
docs/                  troubleshooting notes
```

## License

Released under the MIT License (see `LICENSE`).
