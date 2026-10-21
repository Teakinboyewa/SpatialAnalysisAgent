---
title: Video demonstrations
description: Screen recordings of GIS Copilot running spatial analyses in QGIS, covering vector selection, contours, NDVI, accessibility, terrain, and geometry tasks.
---

# Video demonstrations

<p class="lead">Watch GIS Copilot go from a typed request to finished layers in QGIS. Each recording shows the full run: the request, tool selection, generated code, and the result on the map.</p>

## Quick clips

Short recordings without narration. Each shows one request from start to finish.

<div class="video-grid">
  <div>
    <div class="video">
      <video controls preload="none" poster="assets/img/demo-residential.webp">
        <source src="https://github.com/user-attachments/assets/4c69d024-22c4-4458-ad6f-9b660715aef9" type="video/mp4">
        <a href="https://github.com/user-attachments/assets/4c69d024-22c4-4458-ad6f-9b660715aef9">Download the video</a>
      </video>
    </div>
    <h3>Select residential areas</h3>
    <p><em>"Can you please select all residential areas?"</em> run on a Cambridge land use layer. The agent picks <code>native:extractbyattribute</code> and adds a <code>ResidentialAreas</code> layer. 24 s.</p>
  </div>
  <div>
    <div class="video">
      <video controls preload="none" poster="assets/img/demo-contours.webp">
        <source src="https://github.com/user-attachments/assets/4b23eba4-3e99-47ec-85d6-6efb7ebb6b20" type="video/mp4">
        <a href="https://github.com/user-attachments/assets/4b23eba4-3e99-47ec-85d6-6efb7ebb6b20">Download the video</a>
      </video>
    </div>
    <h3>Contour lines from a DEM</h3>
    <p><em>"Generate contour lines based on the DEM using interval of 2. Add the elevation as an attribute to the contour line layer."</em> The agent selects <code>gdal:contour</code>. 39 s.</p>
  </div>
</div>

## Full walkthroughs

Longer demonstrations from the [GIBD Lab YouTube channel](https://www.youtube.com/@GIBD_PennState).

<div class="video-grid">
  <div>
    <div class="video"><iframe src="https://www.youtube-nocookie.com/embed/wSEQILjgNWI" title="GIS Copilot: Fast food accessibility analysis" loading="lazy" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture; fullscreen" allowfullscreen></iframe></div>
    <h3>Fast food accessibility analysis</h3>
    <p>Counting restaurants per county, scoring accessibility, mapping it, and correlating with obesity. <a href="examples.html#tutorial-4-fast-food-accessibility-and-obesity">Read the tutorial</a>.</p>
  </div>
  <div>
    <div class="video"><iframe src="https://www.youtube-nocookie.com/embed/1QbvKbWEgX0" title="GIS Copilot: Normalized Difference Vegetation Index (NDVI) generation with remote sensing images" loading="lazy" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture; fullscreen" allowfullscreen></iframe></div>
    <h3>NDVI generation from remote sensing images</h3>
    <p>Working out NIR and Red bands from Landsat files, clipping, and running the raster calculator. <a href="examples.html#tutorial-5-ndvi-from-landsat-imagery">Read the tutorial</a>.</p>
  </div>
  <div>
    <div class="video"><iframe src="https://www.youtube-nocookie.com/embed/rhm_wAhPbRU" title="GIS Copilot: Terrain analysis with DEM data" loading="lazy" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture; fullscreen" allowfullscreen></iframe></div>
    <h3>Terrain analysis with DEM data</h3>
    <p>Deriving slope, aspect, hillshade, and other terrain indices from elevation data.</p>
  </div>
  <div>
    <div class="video"><iframe src="https://www.youtube-nocookie.com/embed/rmKfJBOOm6E" title="GIS Copilot: School sidewalk length analysis" loading="lazy" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture; fullscreen" allowfullscreen></iframe></div>
    <h3>School sidewalk length analysis</h3>
    <p>A vector task that measures sidewalk length for school areas.</p>
  </div>
  <div>
    <div class="video"><iframe src="https://www.youtube-nocookie.com/embed/MV2R5cEfpxg" title="GIS Copilot: Geometry calculation and data extraction" loading="lazy" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture; fullscreen" allowfullscreen></iframe></div>
    <h3>Geometry calculation and data extraction</h3>
    <p>Computing geometry attributes and extracting features that meet a condition.</p>
  </div>
</div>

<div class="callout note" markdown="1">
<span class="callout-title">Recorded on earlier releases</span>
Some recordings show older tab names (*Solution Graph* instead of *Geoprocessing Workflow*) and no *Generated Code* tab. The workflow is the same in version 1.4.
</div>
