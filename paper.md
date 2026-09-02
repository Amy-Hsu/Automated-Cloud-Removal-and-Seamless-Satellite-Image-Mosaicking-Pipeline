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
  - name: Chun-Lin Liu
    affiliation: 3
  - name: Chun-Chieh Lo
    affiliation: 3
  - name: Joel Moortgat
    affiliation: 1
affiliations:
  - name: School of Earth Sciences, The Ohio State University, Columbus, OH, USA
    index: 1
  - name: Center for Space and Remote Sensing Research, National Central University, Taoyuan, Taiwan
    index: 2
  - name: National Space Organization, National Applied Research Laboratories, Hsinchu, Taiwan
    index: 3
date: 2 September 2026
bibliography: paper.bib
---

# Summary

Generating seamless, cloud-free mosaics from satellite imagery is a fundamental
task in Earth observation, yet it remains challenging due to radiometric
inconsistencies between scenes, geometric misalignment, and pervasive cloud
contamination. This package provides an end-to-end, largely automated pipeline
that transforms individual orthorectified satellite scenes into a single
color-balanced, seamless, cloud-filled mosaic for both multispectral (XS) and
panchromatic (Pan) products, together with a mosaic-wide cloud mask.

The pipeline consists of four steps: (1) radiometric normalization (color
balance) of each raw scene against reference imagery; (2) automatic seamline
detection via a sparse-node Dijkstra's algorithm over an image-difference cost
surface, followed by seamless mosaicking with Gaussian feathered blending;
(3) panchromatic mosaicking reusing the seamlines found in Step 2; and
(4) automatic cloud filling from a candidate-scene database using
structural-similarity (SSIM) screening and Poisson-style blending.

# Statement of need

Existing satellite mosaicking tools either require extensive manual
intervention (e.g., selecting seamlines by hand in commercial GIS software) or
address only a subset of the problem (e.g., color balancing without cloud
removal). Fully automated, open-source pipelines that handle the complete
workflow from raw scenes to cloud-free mosaics are scarce.

This pipeline fills that gap. Originally developed and validated for
island-wide SPOT-6/7 mosaics of Taiwan, it is designed to be applicable to
other orthorectified optical imagery in any region by adapting the
configuration parameters. The modular four-step design allows users to run only
the steps they need or to substitute individual components.

Key algorithmic contributions include:

- **Sparse-node Dijkstra seamline optimization**: rather than building a full
  pixel-level graph, the algorithm downsamples the overlap region and finds
  cheapest-cost paths along natural features (roads, rivers, field boundaries),
  dramatically reducing computation while producing visually superior seams.
- **Geometric-aware boundary intersection**: automatically computes the true
  geometric intersection of irregularly shaped image swaths to determine
  optimal seamline start and end points.
- **SSIM-guided cloud replacement**: for each cloud patch, candidate scenes are
  ranked by structural similarity to the surrounding cloud-free mosaic area,
  ensuring color and texture consistency in the filled result.

# Acknowledgements

The boundary-mask extraction tools (`GetBoundaryMask.exe`,
`BndPolygonize_v.bat`) and the ERDAS IMAGINE spatial models shipped in
`step1_color_balance/` were originally developed at the Center for Space and
Remote Sensing Research (CSRSR), National Central University, Taiwan, and at
the National Space Organization (NSPO/TASA), Taiwan. We gratefully acknowledge
their contributions.

# References
