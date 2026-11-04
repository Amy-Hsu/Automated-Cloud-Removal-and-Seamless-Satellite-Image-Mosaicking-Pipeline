---
title: "Step 2 — Seamless multispectral mosaic"
---

[View this page on GitHub](https://github.com/buckai-observatory/Automated-Cloud-Removal-and-Seamless-Satellite-Image-Mosaicking-Pipeline/blob/main/step2_mosaic/README.md){.github-link}

See the root `README.md` for the full folder layout and `para.txt` reference.

Quick start: place the color-balanced `.img` scenes, `para.txt`, `NSPO_GRID_vectorization/`
(with `GetBoundaryMask.exe` — requires MATLAB Runtime v9.4 — and `BndPolygonize_v.bat`) and the
`cloud_masks/` cloud-mask folder next to `step2mosaic.py`, then run `run.bat`.

Note: the cloud-mask folder name is set by the `CLOUD_MASK_FOLDER` constant near the top of `step2mosaic.py`.

Outputs: `mosaic_xs.tif`, `mosaic_cloudmask.tif`, `ProcessOrder.txt`, `transform.npy`, `allWindow.npy`,
`ImgInd.npy`, `Union_gt.npy`, and per-pair seamline/blend rasters — several of these are
required inputs for Steps 3 and 4.
