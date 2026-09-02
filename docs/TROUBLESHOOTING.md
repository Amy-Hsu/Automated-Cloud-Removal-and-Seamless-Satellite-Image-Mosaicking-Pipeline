# Troubleshooting

Known failure modes collected while running the pipeline operationally.

## General

- **Never** use spaces in any folder or file path.
- When re-running Step 2, delete all previously generated outputs first (`MosTai.tif`, `TaiMas.tif`, `*.npy`, `ProcessOrder.txt`, seamline files, `res_union_new.*`), otherwise the run fails. Files inside `NSPO_GRID_vectorization/` can usually stay.
- When re-running Step 3, delete `MosTai_Pan.tif` first. Clear `new_pan/` only if the source Pan imagery changed.

## Step 2: intersection goes awry

**Symptom:** intersection points cannot be computed, or are wrong.
**Cause:** pixels that should be `0` (outside the footprint) are not `0` — stray values along the scene border confuse the boundary polygonization.
**Fix:** clean the scene borders before mosaicking; verify with the identify tool in QGIS/ArcMap that the area outside the footprint is exactly `0` in all bands.

## Step 2: `No road from target to endnode`

**Symptom:** the seamline search cannot connect the two boundary intersection points.

Case A — the shrunken overlap mask breaks into disconnected pieces (visible as a gap when plotting the mask). Two remedies:

1. Reduce `kernel` in `para.txt`, e.g. `61 → 5`.
2. Enlarge the search tolerance in the code (`pix+30 → pix+500`).

Case B — the overlap region between the two scenes is extremely thin (a sliver). Such pairs are error-prone and produce a visible dark line at the seam even when they succeed. **Recommendation:** don't mosaic pairs with such minimal overlap; choose scenes with healthier overlap instead.

## Step 3: wrong Pan alignment

The Pan scene extents must match their XS counterparts exactly, or the blend is placed incorrectly. Also remove ERDAS side-car files (`.aux`, `.rrd`, `.rde`) from `raw_pan/` before running.

## Step 4: filled patches look displaced / smeared

**Symptom:** after cloud filling, the filled areas look wrong or shifted.
**Cause:** the color-balance step can slightly crop a scene, so the coordinates looked up during cloud processing no longer match.
**Fix:** before running Step 4, overlay each color-balanced scene on its raw original and confirm the extents match; re-do color balance for cropped scenes.

## Step 4: repeated cloud filling

Running Step 4 again with fresh candidates may or may not improve the result, but it is safe to try: update `cloud_database/`, `raw_pan/`, and the cloud-mask database (used scenes can be removed to save compute time). Only mask value `1` in `TaiMas.tif` is processed — set values back to `1` manually for areas you want reprocessed; already-filled areas carry the value `(loop_index+1)*10`.
