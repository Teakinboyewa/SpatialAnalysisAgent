---
title: Step-by-step tutorials
description: Five worked GIS Copilot tutorials from basic to advanced. Each covers the data, the exact request, the selected tools, the workflow graph, the generated code, the result, and how to verify it.
---

# Step-by-step tutorials

<p class="lead">Five worked analyses, from a one-tool selection to an open-ended accessibility study. Each one shows what to type, what GIS Copilot does, the code it writes, and how to check the result.</p>

All the datasets come from the published case studies. Download them from the [test case page](https://giscience.psu.edu/copilot_test/) or see the [case study PDF](https://github.com/Teakinboyewa/SpatialAnalysisAgent/raw/master/Doc/Case%20Studies/GIS_Copilot_Case_Studies.pdf). You can also substitute your own data with the same structure.

| # | Tutorial | Level | Data type | Key tools |
| --- | --- | --- | --- | --- |
| 1 | [Select counties by population](#tutorial-1-select-counties-by-population) | <span class="badge l1">Basic</span> | Vector | `native:extractbyattribute` |
| 2 | [Clip a national land cover raster to a state](#tutorial-2-clip-a-land-cover-raster-to-a-state) | <span class="badge l1">Basic</span> | Raster + vector | `gdal:cliprasterbymasklayer` |
| 3 | [Mean elevation per county choropleth](#tutorial-3-mean-elevation-per-county-choropleth) | <span class="badge l2">Intermediate</span> | Raster + vector | clip, `native:zonalstatisticsfb`, thematic map |
| 4 | [Fast food accessibility and obesity](#tutorial-4-fast-food-accessibility-and-obesity) | <span class="badge l3">Advanced</span> | Vector + table | spatial join, field calculator, seaborn |
| 5 | [NDVI from Landsat imagery](#tutorial-5-ndvi-from-landsat-imagery) | <span class="badge l3">Advanced</span> | Multiband raster | reproject, clip, `qgis:rastercalculator` |

Results vary between models, providers, and QGIS versions. Your code won't match these runs line for line. What matters is that it's correct.

## Tutorial 1: Select counties by population

<span class="badge l1">Basic</span> One tool, one layer. A good first test of field names and operators.

### Data

| Layer | Type | Fields used |
| --- | --- | --- |
| `Contiguous_US_Counties` | Polygon | `Population` (integer) |

Open the attribute table first and confirm that `Population` is **numeric**. If it's stored as text, comparisons such as `> 50000` behave unexpectedly.

### Request

> Select the USA counties that have more than 50,000 population.

### What GIS Copilot does

```text
AI: Selected tool(s): ['native:extractbyattribute']
AI: Geoprocessing workflow is ready.
AI: Done
```

The agent read the field list, matched "population" to the `Population` field, and mapped "more than" to operator `2` (`>`):

```python
parameters = {
    'INPUT': input_layer,
    'FIELD': 'Population',
    'OPERATOR': 2,          # '>' operator
    'VALUE': 50000,
    'OUTPUT': output_path,
}
result = processing.run("native:extractbyattribute", parameters)

# Load the selected features as a new layer in QGIS
output_layer = QgsVectorLayer(result['OUTPUT'], 'Counties_Above_50000', 'ogr')
QgsProject.instance().addMapLayer(output_layer)
```

### Result

<figure>
  <img src="assets/img/populous-counties.webp" alt="US county map with counties above 50,000 population in yellow over purple. The attribute table of Counties_Above_50000 lists 979 features." loading="lazy">
  <figcaption><code>Counties_Above_50000</code> (yellow) over all counties (purple). The attribute table reports 979 features.</figcaption>
</figure>

### Verify

- [ ] Sort the output's attribute table by `Population`. The smallest value should be **above** 50 000, not equal to it.
- [ ] Cross-check the count with *Select by Expression* `"Population" > 50000` on the original layer.

### Try next

- "…and highlight the largest and the smallest county by area in South Carolina." (see [gallery](gallery.md#basic))
- "…then compute the total population of the selected counties and print it."

## Tutorial 2: Clip a land cover raster to a state

<span class="badge l1">Basic</span> A raster-by-vector operation where the defaults matter: NoData, resolution, and the colour palette.

### Data

| Layer | Type | Notes |
| --- | --- | --- |
| US land cover | Paletted raster | National classification with a categorical legend |
| `PA_boundary` | Polygon | Pennsylvania outline |

### Request

> Clip the land cover data of the USA to the Pennsylvania boundary.

### What GIS Copilot does

It selects `gdal:cliprasterbymasklayer` and fills in the GDAL options. It crops to the cutline and keeps the source resolution:

```python
parameters = {
    'INPUT': landcover_layer,
    'MASK': pa_boundary_layer,
    'CROP_TO_CUTLINE': True,
    'KEEP_RESOLUTION': True,
    'SET_RESOLUTION': False,
    'MULTITHREADING': False,
    'OUTPUT': output_raster_path,
    'DATA_TYPE': 0,          # Use input layer data type
}
result = processing.run("gdal:cliprasterbymasklayer", parameters)
```

### Result

<figure>
  <img src="assets/img/pa-landcover-clip.webp" alt="Pennsylvania land cover clipped from the national dataset with the categorical legend preserved. A second map view shows the full US land cover with Pennsylvania outlined in red." loading="lazy">
  <figcaption>The clipped raster keeps its categorical legend (Open Water, Developed, Deciduous Forest, and so on). The inset map shows the national source with the clip boundary.</figcaption>
</figure>

### Verify

- [ ] **CRS:** the raster and boundary should share a CRS. If they don't, ask for a reprojection of the boundary first.
- [ ] **Pixel size:** *Layer Properties → Information* should show the same resolution as the source.
- [ ] **Data type:** a categorical raster must stay integer. `DATA_TYPE: 0` keeps the input type.
- [ ] **Edges:** zoom to the border. Cells outside the boundary should be NoData, not 0.

## Tutorial 3: Mean elevation per county choropleth

<span class="badge l2">Intermediate</span> Three chained operations given as numbered steps. This tutorial shows the full anatomy of a generated script.

### Data

| Layer | Type | Notes |
| --- | --- | --- |
| `PA_DEM` | Single-band raster | Elevation in metres |
| `PA_boundary` | Polygon | State outline used as the clip mask |
| `PennsylvaniaCounties` | Polygon | 67 counties |

### Request

> Perform the following task: 1) Clip the DEM to PA boundaries and load the clipped DEM. 2) Generate the zonal statistics for every county in PA, focusing on average elevation. 3) Create a choropleth map showing the average elevation of each county in PA, using a red color gradient to represent elevation differences. Load the choropleth map.

### Tool selection and plan

```text
AI: Selected tool(s): ['gdal:cliprasterbymasklayer', 'native:zonalstatisticsfb', 'thematic_map_creation']
AI: Graph is ready.
AI: Done
```

<figure class="medium">
  <img src="assets/img/pa-elevation-graph.webp" alt="Workflow graph: load_DEM → PA_DEM.tif and load_PA_boundary → PA_boundary.shp feed clip_DEM_by_PA_boundary → clipped_DEM; clipped_DEM and PennsylvaniaCounties.shp feed zonal_statistics_avg_elevation → avg_elevation_by_county → create_choropleth_map → choropleth_map." loading="lazy">
  <figcaption>The plan. Two inputs meet at the clip, and the clipped DEM joins the counties at the zonal statistics step.</figcaption>
</figure>

### The generated script, annotated

This is the script from the run above, lightly condensed, with paths shortened and comments added. Your script will contain the real paths of your layers and workspace.

```python
import processing
import os
from qgis.core import (QgsVectorLayer, QgsRasterLayer, QgsProject, QgsSymbol,
                       QgsGraduatedSymbolRenderer, QgsGradientColorRamp)
from PyQt5.QtGui import QColor

def perform_dem_analysis():
    # Inputs come from the layers you ticked in the Layers panel
    dem_path = ".../Pennsylvania/PA_DEM.tif"
    counties_path = ".../Pennsylvania/PennsylvaniaCounties.shp"
    boundary_path = ".../Pennsylvania/PA_boundary.shp"

    # Outputs go to the Workspace Directory from Settings
    output_dir = r".../Default_workspace"

    # Never overwrite: append _1, _2, ... if a file already exists
    clipped_dem_path = os.path.join(output_dir, "Clipped_DEM.tif")
    zonal_stats_output_path = os.path.join(output_dir, "Zonal_Statistics.shp")
    count = 1
    while os.path.exists(clipped_dem_path):
        clipped_dem_path = os.path.join(output_dir, f"Clipped_DEM_{count}.tif")
        count += 1
    count = 1
    while os.path.exists(zonal_stats_output_path):
        zonal_stats_output_path = os.path.join(output_dir, f"Zonal_Statistics_{count}.shp")
        count += 1

    # Step 1: Clip the DEM to PA boundaries
    processing.run("gdal:cliprasterbymasklayer", {
        'INPUT': dem_path,
        'MASK': boundary_path,
        'CROP_TO_CUTLINE': True,
        'OUTPUT': clipped_dem_path,
    })
    QgsProject.instance().addMapLayer(QgsRasterLayer(clipped_dem_path, 'Clipped DEM'))

    # Step 2: Zonal statistics, mean elevation per county
    result = processing.run("native:zonalstatisticsfb", {
        'INPUT': counties_path,
        'INPUT_RASTER': clipped_dem_path,
        'RASTER_BAND': 1,
        'COLUMN_PREFIX': 'elev_',
        'STATISTICS': [2],          # 2 = mean
        'OUTPUT': zonal_stats_output_path,
    })

    # Step 3: Choropleth with a light-to-dark red ramp, 5 quantile classes
    layer = QgsVectorLayer(zonal_stats_output_path, "Elevation Choropleth", "ogr")
    QgsProject.instance().addMapLayer(layer)
    renderer = QgsGraduatedSymbolRenderer('', [])
    renderer.setClassAttribute('elev_mean')
    renderer.setMode(QgsGraduatedSymbolRenderer.Quantile)
    renderer.updateClasses(layer, 5)
    renderer.updateColorRamp(QgsGradientColorRamp(QColor(255, 200, 200), QColor(139, 0, 0)))
    layer.setRenderer(renderer)
    layer.triggerRepaint()

perform_dem_analysis()
```

Things worth noticing:

- `STATISTICS: [2]` is QGIS's code for **mean**. Your words "average elevation" were translated into it.
- The new field is named `elev_` + `mean`, giving `elev_mean`, which the renderer then uses.
- "Red colour gradient" became an explicit two-colour `QgsGradientColorRamp`.

### Result

<figure>
  <img src="assets/img/pa-elevation-choropleth.webp" alt="Pennsylvania counties shaded in five red classes of mean elevation from 38 to 673 metres above the clipped DEM. The panel highlights the selected tools." loading="lazy">
  <figcaption><code>Elevation Choropleth</code> in five quantile classes (38–673 m), with the clipped DEM in a second map view.</figcaption>
</figure>

### Verify

- [ ] Use the *Identify* tool on a county and compare `elev_mean` with the DEM values you see there.
- [ ] Check for counties with `NULL` means. That points to polygons outside the clipped raster.
- [ ] Remember the units. `elev_mean` is in the DEM's vertical units (metres here).

### Try next

The South Carolina version of this task adds a histogram, which appears in **Reports**:

> 1) Clip the DEM to SC boundary and load the clipped DEM. 2) Show me the histogram for the clipped DEM pixel values. 3) Generate the zonal statistics for every county in SC, focusing on average elevation. 4) Finally create a choropleth map showing the average elevation of each county in SC.

<figure>
  <img src="assets/img/sc-zonal-stats.webp" alt="South Carolina mean elevation choropleth, DEM histogram in the Reports tab, generated code, and branching workflow graph with histogram and choropleth branches." loading="lazy">
  <figcaption>The four-step SC variant. Note that the plan branches into a histogram path and a choropleth path from the same clipped DEM.</figcaption>
</figure>

## Tutorial 4: Fast food accessibility and obesity

<span class="badge l3">Advanced</span> A goal-oriented request. GIS Copilot has to design the method itself: counting, normalising, mapping, and a statistical plot.

<div class="video">
  <iframe src="https://www.youtube-nocookie.com/embed/wSEQILjgNWI" title="GIS Copilot: Fast food accessibility analysis" loading="lazy" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture; fullscreen" allowfullscreen></iframe>
</div>

### Data

| Layer | Type | Fields used |
| --- | --- | --- |
| `PA_Fastfoods_XY` | Point | One point per restaurant |
| `PennsylvaniaCounties` | Polygon | `Population`, `Obesity` (percent) |

### Request

> Could you analyze and visualize the fast food accessibility score for each county based on the number of fast food restaurants and population using a thematic map with blue graduated colors. Then, analyze the correlation between the county-level obesity rate and the fast food accessibility score by drawing a scatter plot with a regression line.

### What GIS Copilot does

```text
AI: Selected tool(s): ['thematic_map_creation', 'seaborn']
AI: Geoprocessing workflow is ready.
AI: .../ff_obesity_correlation.png
AI: Done
```

The plan it produced:

1. **Spatial join** of restaurants to counties, then count per county.
2. **Accessibility score** = restaurants ÷ population.
3. **Thematic map** of the score with a light-to-dark blue graduated ramp.
4. **Scatter plot** of score vs. obesity rate with a seaborn regression line, saved as PNG and shown in **Reports**.

The styling step of the generated code:

```python
symbol = QgsSymbol.defaultSymbol(result_layer.geometryType())
renderer = QgsGraduatedSymbolRenderer("", [])
renderer.setClassAttribute("FF_Score")
renderer.setMode(QgsGraduatedSymbolRenderer.Quantile)
renderer.updateClasses(result_layer, 5)
color_ramp = QgsGradientColorRamp(QColor(173, 216, 230),   # light blue
                                  QColor(0, 0, 139))       # dark blue
renderer.updateColorRamp(color_ramp)
result_layer.setRenderer(renderer)
result_layer.triggerRepaint()
```

### Result

<figure>
  <img src="assets/img/pa-fastfood-accessibility.webp" alt="Pennsylvania counties in five blue classes of fast food accessibility with restaurant points; a scatter plot of accessibility score vs obesity rate with a red regression line in Reports; the generated code; and a workflow graph from load_fast_food_locations to obesity_correlation_analysis." loading="lazy">
  <figcaption>Map, scatter plot (Reports), code, and workflow graph from one request. The graph shows the agent's own method design: spatial join → count → score → map and correlation.</figcaption>
</figure>

### Verify

- [ ] **Method choices.** The agent defined "accessibility" as count ÷ population. Decide whether that is the definition you want (per 1 000 people? per km²?). If not, state it in the request.
- [ ] **Join correctness.** Spot-check a county: count the points inside it and compare with the computed count.
- [ ] **Statistics.** The plot shows a trend line, not a significance test. Ask for "Pearson r and p-value" if you need numbers.

<div class="callout note" markdown="1">
<span class="callout-title">Want more control?</span>
The same analysis written as numbered steps is shown in the [interface screenshot](user-guide.md): count, then `Score = Count / Population * 1000`, then map. Explicit steps make the method reproducible and easier to review.
</div>

## Tutorial 5: NDVI from Landsat imagery

<span class="badge l3">Advanced</span> Remote sensing with a vague request. The agent must work out which bands are NIR and Red, align them, and clip to the study area.

<div class="video">
  <iframe src="https://www.youtube-nocookie.com/embed/1QbvKbWEgX0" title="GIS Copilot: NDVI generation with remote sensing images" loading="lazy" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture; fullscreen" allowfullscreen></iframe>
</div>

### Data

| Layer | Type | Notes |
| --- | --- | --- |
| `LC08_L1TP_190055_2020021…` | Landsat 8 bands | Separate GeoTIFF per band (B4 = Red, B5 = NIR) |
| `Akure` | Polygon | City boundary used to clip |

### Request

> Generate the Normalized Difference Vegetation Index (NDVI) of Akure from these satellite imageries.

### What GIS Copilot does

```text
AI: Selected tool(s): ['qgis:rastercalculator', 'native:reprojectlayer', 'gdal:cliprasterbymasklayer']
AI: Geoprocessing workflow is ready.
AI: Done
```

It reprojects the boundary to match the imagery, clips the NIR and Red bands, and then applies the NDVI formula with the raster calculator:

```python
parameters_ndvi = {
    'EXPRESSION': '("NIR Clipped@1" - "Red Clipped@1") / ("NIR Clipped@1" + "Red Clipped@1")',
    'LAYERS': [nir_clipped_layer, red_clipped_layer],
    'EXTENT': nir_clipped_layer.extent(),
    'CELL_SIZE': nir_clipped_layer.rasterUnitsPerPixelX(),
    'OUTPUT': ndvi_output,
}
processing.run("qgis:rastercalculator", parameters_ndvi)
ndvi_layer = QgsRasterLayer(ndvi_output, 'NDVI Layer')
QgsProject.instance().addMapLayer(ndvi_layer)
```

### Result

<figure>
  <img src="assets/img/akure-ndvi.webp" alt="NDVI raster of Akure in a viridis ramp from 0.038 to 0.267, with the source Landsat scene and the red Akure boundary in a second view." loading="lazy">
  <figcaption>NDVI for Akure (0.04–0.27). The lower view shows the source scene and the clip boundary.</figcaption>
</figure>

### Verify

- [ ] **Band assignment.** Open the code and confirm that NIR is B5 and Red is B4 for Landsat 8. Swapping them flips the sign of NDVI.
- [ ] **Value range.** NDVI must fall within −1 to 1. Values outside that range point to integer arithmetic or wrong bands.
- [ ] **Reflectance.** Level-1 digital numbers give a *relative* NDVI. For comparisons across dates, ask for reflectance-scaled bands.

<div class="callout tip" markdown="1">
<span class="callout-title">Make it unambiguous</span>
"Compute NDVI = (B5 − B4) / (B5 + B4) using the Landsat 8 files ending `_B5.TIF` and `_B4.TIF`, as floating point. Clip to `Akure` and save as `akure_ndvi.tif`." removes all guesswork.
</div>

## Lesson: always check the numbers

<figure>
  <img src="assets/img/residential-landuse.webp" alt="Cambridge land use map with residential polygons highlighted in dark red. The log shows a first run reporting 0.0 square kilometres, and a second run reporting 10.446 square kilometres, boxed in red." loading="lazy">
  <figcaption>The same request run twice. The first run reported <strong>0.0 km²</strong> without raising any error. The rerun reported <strong>10.446 km²</strong>.</figcaption>
</figure>

The first run of *"select the residential area and calculate the total area in square kilometers"* completed "successfully" with a total of 0. Typical causes of this kind of result are an area computed in a geographic CRS (square degrees), a filter value that matched nothing, or a sum taken before the selection was applied. Nothing crashed, so no debugging was triggered. A quick sanity check exposed the problem: a city's residential land cannot be 0 km².

**Habits that catch this:** state the CRS for area and distance, ask for intermediate counts ("print the number of selected polygons"), and compare one value against a manual QGIS measurement.

## More examples

- [Case study gallery](gallery.md): 19 published runs with prompts and tools.
- [Video demonstrations](videos.md): seven screen recordings.
- [100+ test cases](https://giscience.psu.edu/copilot_test/) and [further demonstrations](https://giscience.psu.edu/gis-copilot-demonstrations/) on the GIBD website.
