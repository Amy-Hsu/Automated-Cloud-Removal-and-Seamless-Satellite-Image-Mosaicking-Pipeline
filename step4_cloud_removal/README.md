# Step 4 — Automatic cloud filling

See the root `README.md` for the folder layout and the `para_cld.txt` reference.

Inputs: `mosaic_xs.tif` + `mosaic_cloudmask.tif` (Step 2), `mosaic_pan.tif` (Step 3),
`allWindow.npy` + `transform.npy` + `ProcessOrder.txt` (Step 2), a user-provided
land/sea border mask (set via `seaBorderAddress` in `para_cld.txt` — must share the
XS grid/alignment), a `cloud_database/` of candidate XS scenes, the matching
cloud-mask database folder, a `raw_pan/` of candidate Pan scenes, and an empty
working folder `cldDatabase/`.

Run `run.bat`. The script fills clouds in place. In `mosaic_cloudmask.tif`, filled
clouds become `(loop_index+1)*10`; unprocessed clouds stay `1`.
