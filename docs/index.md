---
title: GIS Copilot documentation
description: Use natural language to run spatial analysis inside QGIS. Install, configure, and learn GIS Copilot through tutorials, case studies, and video demonstrations.
---

<div class="hero">
  <div>
    <p class="eyebrow">QGIS plugin · Spatial Analysis Agent</p>
    <h1>Ask a spatial question. Get a working QGIS workflow.</h1>
    <p class="lead">GIS Copilot reads your layers, plans the analysis, picks QGIS Processing tools and Python libraries, writes the code, and runs it inside QGIS. The result appears in your project as layers, maps, and charts.</p>
    <div class="actions">
      <a class="button primary" href="installation.html">Install the plugin</a>
      <a class="button secondary" href="quickstart.html">Run your first analysis</a>
    </div>
  </div>
  <figure>
    <img src="assets/img/ui-annotated.webp" alt="QGIS with the GIS Copilot panel docked on the right. A Pennsylvania county map shaded by fast food accessibility fills the canvas. The panel shows the code display, information panel, request box and data paths.">
    <figcaption>GIS Copilot docked in QGIS after a three-step request: count fast food restaurants per county, compute an accessibility score, and draw a thematic map.</figcaption>
  </figure>
</div>

<div class="stats">
  <div class="stat"><b>100+</b><span>published test cases and demonstrations</span></div>
  <div class="stat"><b>3</b><span>difficulty levels, from single tools to open-ended tasks</span></div>
  <div class="stat"><b>Vector + raster</b><span>QGIS Processing, GDAL, GeoPandas, Rasterio, seaborn</span></div>
  <div class="stat"><b>Hosted or local</b><span>OpenAI, GIBD, or an OpenAI-compatible local server</span></div>
</div>

## See it in action

<div class="video">
  <iframe src="https://www.youtube-nocookie.com/embed/wSEQILjgNWI" title="GIS Copilot: Fast food accessibility analysis" loading="lazy" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture; fullscreen" allowfullscreen></iframe>
</div>

In this demo, a single request asks GIS Copilot to calculate a fast food accessibility score for every Pennsylvania county, draw a thematic map, and plot the score against obesity rates. Browse [all video demonstrations](videos.md) for NDVI, terrain, and geometry examples.

## How an analysis runs

<ol class="steps">
  <li><strong>You describe the task.</strong> Check the input layers in the QGIS Layers panel, type the request, and select <em>Send Request</em>.</li>
  <li><strong>GIS Copilot inspects the data.</strong> It reads layer names, geometry types, fields, CRS, and raster bands, then breaks the task into operations.</li>
  <li><strong>It selects tools.</strong> Each operation is matched to a QGIS Processing algorithm (for example <code>native:buffer</code> or <code>gdal:cliprasterbymasklayer</code>) or to a Python library such as seaborn, using the plugin's tool documentation.</li>
  <li><strong>It plans the workflow.</strong> The plan is drawn as a graph of inputs, operations, and outputs in the <em>Geoprocessing Workflow</em> tab.</li>
  <li><strong>It writes and runs Python.</strong> Code is generated, optionally reviewed by the model, and executed in QGIS. On an error, the plugin attempts to debug and retry.</li>
  <li><strong>You inspect the result.</strong> New layers are added to the map, charts appear in <em>Reports</em>, and the final script stays in <em>Generated Code</em> to edit, rerun, or save.</li>
</ol>

<div class="figure-row">
  <figure>
    <img src="assets/img/pa-elevation-graph.webp" alt="Workflow graph: load DEM, load PA boundary, clip DEM by boundary, zonal statistics with counties, create choropleth map." loading="lazy">
    <figcaption><strong>Plan.</strong> Workflow graph for "clip the DEM, compute mean elevation per county, map it".</figcaption>
  </figure>
  <figure>
    <img src="assets/img/sc-zonal-stats.webp" alt="South Carolina county choropleth of mean elevation, a histogram of DEM values in the Reports tab, generated code, and the workflow graph." loading="lazy">
    <figcaption><strong>Result.</strong> Choropleth layer, histogram in Reports, and the code that produced them.</figcaption>
  </figure>
</div>

<div class="callout warning" markdown="1">
<span class="callout-title">Generated code runs automatically</span>
Sending a request executes the generated code immediately. The *Generated Code* tab lets you inspect and rerun code afterwards; it is not an approval step. A "Done" message only means the code ran without error. Always check that the method and the output are what you intended.
</div>

## Where to go next

<div class="cards">
  <a class="card" href="installation.html"><span class="icon">⬇️</span><h3>Install</h3><p>Install from the QGIS catalog or a ZIP and let the plugin set up its Python dependencies.</p></a>
  <a class="card" href="quickstart.html"><span class="icon">🚀</span><h3>First analysis</h3><p>A 10-minute walkthrough: buffer a point layer, then inspect the code, graph, and output.</p></a>
  <a class="card" href="prompts.html"><span class="icon">✍️</span><h3>Write good requests</h3><p>Request templates and before-and-after rewrites that improve reliability.</p></a>
  <a class="card" href="examples.html"><span class="icon">🧭</span><h3>Tutorials</h3><p>Five worked examples with data, prompts, workflow graphs, generated code, and checks.</p></a>
  <a class="card" href="gallery.html"><span class="icon">🗺️</span><h3>Case study gallery</h3><p>Nineteen published runs across basic, intermediate, and advanced tasks.</p></a>
  <a class="card" href="troubleshooting.html"><span class="icon">🛠️</span><h3>Troubleshooting</h3><p>Dependency, model, and execution problems, plus answers to common questions.</p></a>
</div>

## Research and citation

GIS Copilot is developed by the [Geoinformation and Big Data Research Lab (GIBD)](https://giscience.psu.edu/) at Penn State. The framework is described in:

> Temitope Akinboyewa, Zhenlong Li, Huan Ning, and M. Naser Lessani. "GIS Copilot: Towards an Autonomous GIS Agent for Spatial Analysis." *International Journal of Digital Earth*, 2025. [doi:10.1080/17538947.2025.2497489](https://doi.org/10.1080/17538947.2025.2497489)

```bibtex
@article{akinboyewa2025giscopilot,
  title   = {GIS Copilot: Towards an Autonomous GIS Agent for Spatial Analysis},
  author  = {Akinboyewa, Temitope and Li, Zhenlong and Ning, Huan and Lessani, M. Naser},
  journal = {International Journal of Digital Earth},
  year    = {2025},
  doi     = {10.1080/17538947.2025.2497489}
}
```

<div class="callout note" markdown="1">
<span class="callout-title">Use the updated test data links</span>
The full set of test cases and datasets is in the [case study PDF](https://github.com/Teakinboyewa/SpatialAnalysisAgent/raw/master/Doc/Case%20Studies/GIS_Copilot_Case_Studies.pdf) and on the [test case page](https://giscience.psu.edu/copilot_test/). Use these links instead of the ones printed in the published manuscript.
</div>

These pages document plugin **version 1.4**. Release notes are in [metadata.txt](https://github.com/Teakinboyewa/SpatialAnalysisAgent/blob/master/metadata.txt). Screenshots come from several releases, so tab names can differ slightly (older builds call *Geoprocessing Workflow* "Solution Graph").
