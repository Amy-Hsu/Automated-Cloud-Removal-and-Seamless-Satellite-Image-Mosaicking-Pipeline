# Step 3 — Panchromatic mosaic

Copy from Step 2: `transform.npy`, `ImgInd.npy`, `allWindow.npy`, `Union_gt.npy`,
`ProcessOrder.txt`, and all `*MosTai_sl_polygon.img` (+`.ige`) files.
Add `raw_pan/` (one folder per Pan scene containing `A0XXXXXX_B1.DAT.raw`; footprints must
match the XS scenes exactly; delete any `.aux`/`.rrd`/`.rde`) and an empty `new_pan/`.

Configure `para_pan.txt` (`res`, `trueBits`, `StartLetter`, `upscale_factor`) and run `run.bat`.
Output: `MosTai_Pan.tif`. When re-running, delete `MosTai_Pan.tif` first.
