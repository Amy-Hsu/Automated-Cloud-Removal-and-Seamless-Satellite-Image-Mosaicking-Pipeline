# Step 1 — Color balance (ERDAS IMAGINE 2014)

Use **ERDAS IMAGINE 2014** — later versions (2018/2020) may produce wrong colors.
Applies to SPOT-6/7; FS-2/FS-5 also work with different references/parameters (FS-2 needs Step 1 only).

Chain (run everything with `Auto_2data_test.bat` once set up):

1. `Step1SPOTf` (model `color_balance.gmdx`, batch snapshot `SM_BATCH_GMDX_012516`)
   — color balance against invariant areas. Inputs per scene: river reference, Taiwan-border
   reference, reference mosaic, and the raw scene. Outputs: mask, `col1..4_N.tbl`
   (write them into `distrib/`!), and a first balanced image.
2. `distrib/runrunrun.exe` (source: `distrib/runrunrun.py`) — remaps the tables with
   `MappingReshapeXS.exe` so near-zero DN values are stretched (more shadow detail) → `colX_N.fd.tbl`.
3. `Step2SPOTf` (model `applytable_taibor_noedge.gmdx`, snapshot `SM_BATCH_GMDX_023448`)
   — applies the remapped tables → final color-balanced `aXXXXXXX.img`.

Machine setup:

1. Edit output folders in `Step1SPOTf.bcf` / `Step2SPOTf.bcf`; point `smprocess` at the local
   `SM_BATCH_GMDX_*` files. **No spaces in any path.**
2. Create `.bls` scene lists from the `.bls.example` files (tab-separated).
3. ERDAS IMAGINE → Batch: load `.bcf` + `.bls`, submit once to generate
   `Step1SPOTf.bat` / `Step2SPOTf.bat` for your machine (not shipped: they embed local user paths).
4. Afterwards just run `Auto_2data_test.bat` from CMD.
