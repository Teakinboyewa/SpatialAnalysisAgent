---
title: Writing effective requests
description: How to phrase requests that GIS Copilot executes reliably. Covers the anatomy of a good request, templates by task type, and before-and-after rewrites.
---

# Writing effective requests

<p class="lead">GIS Copilot only knows what your layers contain and what you type. A precise request is the single biggest factor in getting a correct result on the first try.</p>

## Anatomy of a good request

A reliable request answers five questions:

| | Question | Example |
| --- | --- | --- |
| 1 | **Which inputs?** Exact layer names, and field names where relevant | `PennsylvaniaCounties`, field `Population` |
| 2 | **Which operation?** The method, not just the goal | "zonal statistics (mean)", not "analyse elevation" |
| 3 | **Which parameters?** Distances *with units*, thresholds, classes, colours | "2000 feet", "> 50 000", "5 quantile classes, blue ramp" |
| 4 | **Which CRS?** When distance or area matters | "reproject to EPSG:26918 (metres) first" |
| 5 | **Which output?** Format, name, and whether to add it to the map | "save as `zones.gpkg` in the workspace and add it to the map" |

Put together:

> Reproject `PA_Fastfoods` to EPSG:26918. Count the points in each polygon of `PennsylvaniaCounties` and store the result in a new field `Count`. Then compute `Score = Count / Population * 1000`. Draw a thematic map of `Score` with 5 quantile classes from light to dark orange. Save the scored layer as `ff_score.gpkg` in the workspace and add it to the map.

## Three levels of guidance

The [case studies](gallery.md) test the agent at three levels. Choose the style that fits your task:

<div class="cards">
  <div class="card"><span class="badge l1">Basic</span><h3>One operation</h3><p>"Generate contour lines from the DEM of Puerto Rico with a 50-meter interval."</p></div>
  <div class="card"><span class="badge l2">Intermediate</span><h3>Numbered steps</h3><p>"1) Clip the DEM to the SC boundary. 2) Show a histogram of pixel values. 3) Zonal statistics (mean) per county. 4) Choropleth map."</p></div>
  <div class="card"><span class="badge l3">Advanced</span><h3>Goal only</h3><p>"Analyse fast food accessibility for each county and its correlation with obesity rate."</p></div>
</div>

**Numbered steps are the most reliable** for multi-step work. Goal-only requests let the agent design the method, which is powerful but makes careful review more important.

## Templates by task

Replace the parts in `<angle brackets>`.

### Select or extract by attribute

> Select features in `<layer>` where `<field>` `<operator>` `<value>`. Save them as a new layer `<name>.gpkg` and add it to the map.

### Buffer

> Create `<distance> <units>` buffers around `<layer>`. `<Dissolve | Keep separate>` overlapping buffers. If the layer CRS is geographic, reproject to `<EPSG code>` first. Save as `<name>.gpkg`.

### Clip a raster

> Clip `<raster>` to `<polygon layer>`. Keep the original resolution and NoData value. Save as `<name>.tif` and add it to the map.

### Zonal statistics

> Calculate the `<mean | sum | max…>` of `<raster>` band `<n>` for each polygon in `<zones layer>`. Use the prefix `<prefix_>` for new fields. Save as `<name>.gpkg`.

### Terrain

> From `<DEM>`, compute slope (degrees), aspect, and hillshade. Clip each to `<boundary>`. Add all outputs to the map.

### Thematic map

> Style `<layer>` as a graduated map of `<field>` with `<n>` `<quantile | equal interval | natural breaks>` classes using a `<colour>` ramp.

### Chart or statistics

> Using `<layer>`, draw a `<scatter plot with regression line | histogram | bar chart>` of `<field(s)>`. Label the axes clearly and save it as a PNG in the workspace.

### Join a table

> Join `<csv>` to `<layer>` using `<csv key>` = `<layer key>`. Then calculate `<new field>` = `<expression>`.

## Before and after

| Vague | Precise |
| --- | --- |
| Buffer the hospitals. | Create 1 km buffers around `hospitals` after reprojecting to EPSG:32618. Dissolve overlaps. Save as `hospital_1km.gpkg`. |
| Show me elevation by county. | Calculate mean elevation from `PA_DEM` for each polygon in `PennsylvaniaCounties`, then draw a 5-class quantile choropleth of the mean in a red ramp. |
| Find big counties. | From `US_Counties`, extract counties where `Population` > 50000 into a new layer `Counties_Above_50000`. |
| Make an NDVI. | Compute NDVI = (NIR − Red) / (NIR + Red) using band 5 (`…_B5.TIF`) as NIR and band 4 (`…_B4.TIF`) as Red. Clip to `Akure` first and save as `ndvi.tif`. |
| Get the residential area. | In `Land_Use`, select polygons where `LandUse` = 'Residential'. Compute their total area in km² after reprojecting to EPSG:26986, and print the total. |

## Tips that matter

- **Be explicit about units.** "2000" alone is ambiguous: degrees, feet, and metres all produce valid-looking results.
- **Name fields exactly as they appear** in the attribute table, including case. If you're not sure, ask first: "List the fields in `US_Counties`."
- **Say "add it to the map"** when you want to see the output. Otherwise it may only be written to disk.
- **Restate context when refining.** Each request is planned from scratch. "Do it again with 500 m" lacks the inputs.
- **Split very long tasks.** If a request has more than five or six operations, run it in two parts and check the intermediate output.
- **Ask for a report** when you need numbers ("print the total area" or "save the table as HTML") so the value appears in the log or the Reports tab.

<div class="callout tip" markdown="1">
<span class="callout-title">Use questions to explore</span>
Requests don't have to change data. "How many features are in `SC_Counties`?" or "Which county has the largest area?" are useful for sanity checks before a heavier analysis.
</div>
