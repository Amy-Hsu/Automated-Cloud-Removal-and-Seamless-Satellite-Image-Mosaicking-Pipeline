# Step 3 — Panchromatic mosaic

Copy from Step 2: `transform.npy`, `ImgInd.npy`, `allWindow.npy`, `Union_gt.npy`,
`ProcessOrder.txt`, and all `*_seamline_polygon.img` (+`.ige`) files.
Add `raw_pan/` (one folder per Pan scene containing the raw `.DAT.raw`; footprints must
match the XS scenes exactly; delete any `.aux`/`.rrd`/`.rde`) and an empty `new_pan/`.

Configure `para_pan.txt` (`res`, `trueBits`, `StartLetter`, `upscale_factor`,
`canvas_crs`, `canvas_x_max`, `canvas_y_min`) and run `run.bat`.
Output: `mosaic_pan.tif`. When re-running, delete `mosaic_pan.tif` first.
