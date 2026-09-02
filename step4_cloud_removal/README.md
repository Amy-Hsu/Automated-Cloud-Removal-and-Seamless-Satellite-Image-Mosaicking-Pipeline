# Step 4 — Automatic cloud filling

See the root `README.md` for the folder layout and the `para_cld.txt` reference.

Inputs: `MosTai.tif` + `TaiMas.tif` (Step 2), `MosTai_Pan.tif` (Step 3),
`allWindow.npy` + `transform.npy` + `ProcessOrder.txt` (Step 2), the land/sea mask
`tw_city6m.img` (user-provided land/sea border mask — must share the XS grid/alignment), a `cloud_database/` of
candidate XS scenes, the matching cloud-mask database folder, a `raw_pan/` of candidate
Pan scenes, and an empty working folder `cldDatabase/`.

Run `run.bat`. The script first copies the input mosaics to `*_Raw.tif`, then fills clouds
in place. In `TaiMas.tif`, filled clouds become `(loop_index+1)*10`; unprocessed clouds stay `1`.
