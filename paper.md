---
title: 'Automated Cloud Removal and Seamless Satellite Image Mosaicking Pipeline'
tags:
  - Python
  - remote sensing
  - satellite imagery
  - cloud removal
  - image mosaicking
  - seamline optimization
authors:
  - name: Hsiao-Jou Hsu
    orcid: 0009-0008-5863-3229
    affiliation: "1, 2"
    corresponding: true
  - name: Kuo-Hsin Tseng
    affiliation: 2
  - name: Fuan Tsai
    affiliation: 2
  - name: Chian-Liang Liu
    affiliation: 2
  - name: Chi-Chuan Lo
    affiliation: 2
  - name: Joachim Moortgat
    affiliation: 1
affiliations:
  - name: School of Earth Sciences, The Ohio State University, Columbus, OH, USA
    index: 1
  - name: Center for Space and Remote Sensing Research, National Central University, Taoyuan, Taiwan
    index: 2
date: 15 October 2026
bibliography: paper.bib
---

# Summary

Optical satellite mosaics are the base layer for national mapping,
environmental monitoring and change detection, but in cloud-prone regions they
are hard to produce: no single acquisition is cloud-free, neighbouring scenes
differ radiometrically, and visible seams appear wherever two scenes meet. This
software is an end-to-end pipeline that turns a set of orthorectified
multispectral (XS) and panchromatic (Pan) scenes into a single color-balanced,
seamless, cloud-filled mosaic, together with a mosaic-wide cloud mask that
records which source filled each cloud.

The pipeline has four steps, each in its own folder and driven by a small JSON
parameter file: (1) radiometric normalization of each scene against reference
imagery using stable, cloud-free pixels; (2) automatic seamline detection with a
shortest-path search over a sparse graph built on the downsampled overlap
region, followed by Gaussian-feathered mosaicking of the XS scenes; (3) Pan
mosaicking that reuses the Step 2 seamlines at Pan resolution; and (4) cloud
filling, in which each cloud patch is replaced from a database of candidate
scenes ranked by structural similarity (SSIM) [@wang2004ssim] to the
surrounding cloud-free mosaic. The methods and their validation are described
in @hsu2026novel; this paper describes the software.

# Statement of need

Producing a large-area, cloud-free mosaic from high-resolution imagery is
normally a manual job: an analyst ranks scenes, tunes color balancing, edits
seamlines by hand and patches clouds one by one in commercial GIS software.
Each of these tasks has a large literature, but the tasks are usually solved in
isolation, and errors in one stage propagate to the next. For example, cloud
pixels that are not masked before color balancing bias the normalization, and
that bias then shows up in the seamline cost surface.

The pipeline targets remote-sensing researchers and mapping agencies who need
reproducible, analysis-ready mosaics from meter-scale optical imagery over
regions where cloud cover is persistent. It was developed for island-wide
mosaics of Taiwan, where cloud frequency exceeds 90% over the central
mountains, and has been applied to SPOT-6/7, Formosat-2/5 and Pléiades imagery
[@hsu2026novel]. Canvas extent, coordinate reference system, bit depth,
resolution and seamline cost function are all parameters, so the same code can
be applied to other sensors and regions. Because each step reads and writes
plain GeoTIFF/NumPy files, users can run only the steps they need or replace
one stage with their own method.

# State of the field

Per-pixel temporal compositing, as popularized by Google Earth Engine
[@gorelick2017gee], is efficient for cloud clearing, but adjacent pixels can
come from different dates. The result is not a physically coherent image of a
single acquisition, and it shows speckle at meter scale. Relative radiometric
normalization methods such as iteratively re-weighted MAD
[@canty2008irmad] address color balance but not seamlines or clouds. Classical
seamline methods formulate the overlap as a dense pixel graph and solve it with
shortest-path search [@dijkstra1959] or graph cuts [@boykov2001graphcuts].
These methods are robust but scale poorly to full-resolution scenes of tens of
millions of pixels. Open-source toolkits such as Orfeo ToolBox
[@grizonnet2017otb] provide mosaicking with feathering and color harmonization.
Like commercial packages, they assume cloud-free, preprocessed inputs and do not
feed cloud information back into seamline placement or scene selection.

We built a new pipeline rather than extending an existing toolkit because the
contribution is how the stages are coupled. Cloud masks drive normalization
statistics, seamline placement and the choice of fill scenes, and the same
seamlines are reused across XS and Pan products. Adding this to a
general-purpose framework would require changing the data model of each stage.
Instead, the pipeline is built on standard Python libraries (rasterio
[@rasterio], NetworkX [@hagberg2008networkx], OpenCV [@bradski2000opencv],
scikit-image [@vanderwalt2014skimage]) and GDAL [@gdal] command-line tools,
so that individual stages remain easy to replace.

# Software design

**Sparse-graph seamline search.** A dense pixel graph over a full-resolution
overlap is too large to search. Step 2 instead downsamples the overlap by a
user-set factor (`resampleTo`, typically 5) and erodes the valid area with a
shrink kernel. It then builds a NetworkX graph only over pixels that are valid
in both scenes. Edge weights are inverse local intensity differences in both
images (four cost functions, `a6`, `a7`, `a6c` and `a7c`, are selectable). The
cheapest path therefore follows strong edges such as rivers, roads and field
boundaries, where a seam is least visible. Start and end nodes come from the
geometric intersection of the two scenes' footprint boundaries, which handles
irregular, non-rectangular swaths. The trade-off is a coarser seamline in
exchange for far fewer graph nodes: reducing 6 m imagery to 30 m cut nodes from
16.5 million to 0.57 million and runtime from 1.42 h to 0.04 h, with near-linear
scaling [@hsu2026novel].

**File-based hand-off between steps.** Each step writes its results (seamline
rasters, per-scene windows and transforms, processing order) to disk instead of
holding them in memory. This lets a long regional run be restarted from any
step. It also lets Step 3 reproduce the XS seamlines at Pan resolution without
recomputing them, which keeps the XS and Pan products geometrically consistent.
All raster I/O uses rasterio windows, so mosaics larger than RAM (for example,
33,417 × 50,893 pixels for Formosat-2) can be written block by block.

**Patch-level, not pixel-level, cloud filling.** Step 4 segments the mosaic
cloud mask into connected patches. For each patch it computes SSIM between
every candidate scene and the cloud-free part of the mosaic window around the
patch, with clouds and sea masked out. Candidates that are empty, already used
at that location, or below the SSIM threshold (0.75 by default) are rejected.
The best remaining candidate that is at least 75% cloud-free under the patch is
inserted with Poisson blending [@perez2003poisson], which takes the texture of
the candidate and the brightness of the surrounding mosaic, so that no seam is
visible. Pixels that are cloudy in the candidate itself are then blended back
with a Gaussian-feathered mask. Using a single real acquisition per patch keeps
the result physically interpretable. This is a deliberate alternative to
generative or per-pixel reconstruction, at the cost of leaving a patch unfilled
when no sufficiently similar, clear candidate exists. Such patches remain
flagged in the output cloud mask. Both thresholds are set in the Step 4
parameter file.

**Limitations.** Step 1 uses spatial models for ERDAS IMAGINE, which is
proprietary. Users without it can supply their own color-balanced scenes and
start at Step 2. Steps 2–4 are Python and run on Windows, macOS and Linux; on
Windows, Step 2 can still use the original compiled boundary-mask tool, and
elsewhere it uses an equivalent Python implementation. Step 4 requires
OpenCV < 4.11, because later versions of `seamlessClone` no longer transfer the
source texture; a unit test guards against this.

# Research impact statement

The pipeline produced the island-wide SPOT-6/7 and Formosat-2 mosaics of Taiwan
reported in @hsu2026novel, built from 14 primary SPOT scenes plus 6
cloud-filling candidates, and 7 Formosat-2 scenes filled from 14 candidate
images. Cloud
removal rates were 97.8% (SPOT) and 89.9% (Formosat-2), and the mean RMSE in
overlapping regions stayed below 11.7% of the 12-bit dynamic range. The same
code also mosaicked Pléiades scenes without modification beyond parameters.
The original imagery cannot be redistributed, so the repository includes a
demonstration that downloads free Sentinel-2 imagery of the same region and
runs Steps 2–4 end to end in a few minutes; on it, Step 4 fills roughly three
quarters of the cloud-covered area. The repository also includes the full
four-step workflow, parameter templates, step-by-step documentation, a
troubleshooting guide and a unit-test suite run in continuous integration on
Linux and Windows.

# AI usage disclosure

Generative AI (Anthropic's Claude) was used to help reorganize the repository,
rename variables and files, write documentation, write the unit tests and
continuous-integration configuration, write the Sentinel-2 demonstration,
make Steps 2–4 portable across operating systems, and draft and edit this
paper. The seamline, normalization and cloud-filling algorithms and their
implementation were developed by H.-J. Hsu. All AI-assisted changes were
reviewed by the authors, checked against the original code and the published
results, and verified by running the test suite and the demonstration.

# Acknowledgements

This research was supported in part by the National Science and Technology
Council of Taiwan (grants 113-2119-M-008-015, 113-2119-M-006-007 and
113-2740-M-008-006). The boundary-mask extraction tools and the ERDAS IMAGINE
spatial models in `step1_color_balance/` were originally developed at the
Center for Space and Remote Sensing Research, National Central University, and
at the Taiwan Space Agency (formerly the National Space Organization).

# References
