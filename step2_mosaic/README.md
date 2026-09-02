# Step 2 — Seamless multispectral mosaic

See the root `README.md` for the full folder layout and `para.txt` reference.

Quick start: place the color-balanced `.img` scenes, `para.txt`, `NSPO_GRID_vectorization/`
(with `GetBoundaryMask.exe` — requires MATLAB Runtime v9.4 — and `BndPolygonize_v.bat`) and the
`cloud_masks/` cloud-mask folder next to `step2mosaic.py`, then run `run.bat`.

Note: the cloud-mask folder name is set by the `CLOUD_MASK_FOLDER` constant near the top of `step2mosaic.py`.

Outputs: `MosTai.tif`, `TaiMas.tif`, `ProcessOrder.txt`, `transform.npy`, `allWindow.npy`,
`ImgInd.npy`, `Union_gt.npy`, and per-pair seamline/blend rasters — several of these are
required inputs for Steps 3 and 4.
