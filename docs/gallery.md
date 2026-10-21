---
title: Case study gallery
description: Nineteen published GIS Copilot runs across basic, intermediate, and advanced spatial analysis tasks, each with the request, the selected tools, and the result.
---

# Case study gallery

<p class="lead">Real runs from the GIS Copilot evaluation. Each card shows the request as typed, the tools the agent selected, and a screenshot of the result. Select any image to enlarge it.</p>

The evaluation groups tasks into three levels:

| Level | What the agent receives | What it must do |
| --- | --- | --- |
| <span class="badge l1">Basic</span> | A direct request, usually one tool and one or two layers | Pick the right tool and parameters |
| <span class="badge l2">Intermediate</span> | Several steps, often numbered | Generate and chain correct code for each step |
| <span class="badge l3">Advanced</span> | A goal, without steps | Design the method, select tools, and execute end to end |

<div class="callout note" markdown="1">
<span class="callout-title">Datasets and the full test set</span>
Datasets for these cases, plus 100+ more test cases, are on the [GIS Copilot test page](https://giscience.psu.edu/copilot_test/) and in the [case study PDF](https://github.com/Teakinboyewa/SpatialAnalysisAgent/raw/master/Doc/Case%20Studies/GIS_Copilot_Case_Studies.pdf). These screenshots span several plugin releases and models, and a rerun today may produce different code.
</div>

## Basic

<div class="gallery">
  <article class="case">
    <figure><img src="assets/img/dc-health-buffer.webp" alt="Green buffers around health facility points within the Washington DC boundary." loading="lazy"><figcaption hidden>Buffers around DC health facilities</figcaption></figure>
    <div class="case-body">
      <span class="badge l1">Basic · vector</span>
      <h3>Health facility service zones</h3>
      <p class="prompt">"Can you please create 2000-feet zones around each health facilities in Washington DC to identify areas of service coverage?"</p>
      <p class="tools">Tools: <code>native:buffer</code></p>
    </div>
  </article>
  <article class="case">
    <figure><img src="assets/img/pr-contours.webp" alt="Contour lines over a Puerto Rico DEM in a yellow-purple ramp." loading="lazy"><figcaption hidden>Contours from the Puerto Rico DEM</figcaption></figure>
    <div class="case-body">
      <span class="badge l1">Basic · raster</span>
      <h3>Contours from a DEM</h3>
      <p class="prompt">"Generate contour lines from the DEM of Puerto Rico with a 50-meter interval."</p>
      <p class="tools">Tools: <code>gdal:contour</code></p>
    </div>
  </article>
  <article class="case">
    <figure><img src="assets/img/populous-counties.webp" alt="US counties above 50,000 population highlighted in yellow." loading="lazy"><figcaption hidden>Counties above 50,000 population</figcaption></figure>
    <div class="case-body">
      <span class="badge l1">Basic · vector</span>
      <h3>Attribute selection</h3>
      <p class="prompt">"Select the USA counties that have more than 50,000 population."</p>
      <p class="tools">Tools: <code>native:extractbyattribute</code> · <a href="examples.html#tutorial-1-select-counties-by-population">Tutorial</a></p>
    </div>
  </article>
  <article class="case">
    <figure><img src="assets/img/pa-landcover-clip.webp" alt="Pennsylvania land cover clipped from US data, with categorical legend." loading="lazy"><figcaption hidden>Land cover clipped to Pennsylvania</figcaption></figure>
    <div class="case-body">
      <span class="badge l1">Basic · raster</span>
      <h3>Clip land cover to a state</h3>
      <p class="prompt">"Clip the land cover data of the USA to the Pennsylvania boundary."</p>
      <p class="tools">Tools: <code>gdal:cliprasterbymasklayer</code> · <a href="examples.html#tutorial-2-clip-a-land-cover-raster-to-a-state">Tutorial</a></p>
    </div>
  </article>
  <article class="case">
    <figure><img src="assets/img/sc-largest-smallest.webp" alt="South Carolina counties with the largest county in blue and the smallest in green." loading="lazy"><figcaption hidden>Largest and smallest SC county</figcaption></figure>
    <div class="case-body">
      <span class="badge l1">Basic · vector</span>
      <h3>Largest and smallest county</h3>
      <p class="prompt">"Please highlight the largest county and the smallest county in South Carolina."</p>
      <p class="tools">Tools: <code>native:calculateexpression</code>, <code>native:extractbyattribute</code></p>
    </div>
  </article>
  <article class="case">
    <figure><img src="assets/img/sf-short-roads.webp" alt="San Francisco road network with short roads extracted and an attribute table with a Length field." loading="lazy"><figcaption hidden>Roads shorter than 100 m</figcaption></figure>
    <div class="case-body">
      <span class="badge l1">Basic · vector</span>
      <h3>Reproject, measure, extract</h3>
      <p class="prompt">"1) Reproject this road vector layer to NAD27 / UTM Zone 10. 2) Create a new field containing the road features' length. Let the name of the field be "Length". 3) Extract roads that have length shorter than 100 meters."</p>
      <p class="tools">Tools: <code>native:reprojectlayer</code>, <code>native:fieldcalculator</code>, <code>native:extractbyexpression</code></p>
    </div>
  </article>
  <article class="case">
    <figure><img src="assets/img/pa-fastfood-elevation.webp" alt="Fast food restaurant points over a Pennsylvania DEM, with an attribute table showing x, y and Elev_1 fields." loading="lazy"><figcaption hidden>Elevation sampled at fast food restaurants</figcaption></figure>
    <div class="case-body">
      <span class="badge l1">Basic · raster + vector</span>
      <h3>Sample elevation at points</h3>
      <p class="prompt">"Add new fields for x and y to the fast food restaurants layer, then extract the elevation values from the DEM layer and save the results in a new layer."</p>
      <p class="tools">Tools: <code>native:addxyfields</code>, <code>native:rastersampling</code></p>
    </div>
  </article>
  <article class="case">
    <figure><img src="assets/img/sc-pop-change.webp" alt="Bar chart of population change from 2000 to 2008 for South Carolina counties in the Reports tab." loading="lazy"><figcaption hidden>SC population change chart</figcaption></figure>
    <div class="case-body">
      <span class="badge l1">Basic · table + chart</span>
      <h3>Population change chart</h3>
      <p class="prompt">A field calculation of population change between 2000 and 2008 for each South Carolina county, plotted as a bar chart.</p>
      <p class="tools">Output: chart in the <strong>Reports</strong> tab</p>
    </div>
  </article>
</div>

## Intermediate

<div class="gallery">
  <article class="case">
    <figure><img src="assets/img/sc-zonal-stats.webp" alt="SC mean elevation choropleth, DEM histogram, code and workflow graph." loading="lazy"><figcaption hidden>South Carolina zonal statistics</figcaption></figure>
    <div class="case-body">
      <span class="badge l2">Intermediate · raster + vector</span>
      <h3>Clip, histogram, zonal statistics, choropleth</h3>
      <p class="prompt">"1) Clip the DEM to SC boundary and load the clipped DEM. 2) Show me the histogram for the clipped DEM pixel values. 3) Generate the zonal statistics for every county in SC, focusing on average elevation. 4) Finally create a choropleth map showing the average elevation of each county in SC."</p>
      <p class="tools">Tools: <code>gdal:cliprasterbymasklayer</code>, <code>seaborn</code>, <code>native:zonalstatisticsfb</code>, thematic map</p>
    </div>
  </article>
  <article class="case">
    <figure><img src="assets/img/pa-elevation-choropleth.webp" alt="Pennsylvania counties shaded by mean elevation in red." loading="lazy"><figcaption hidden>Pennsylvania mean elevation choropleth</figcaption></figure>
    <div class="case-body">
      <span class="badge l2">Intermediate · raster + vector</span>
      <h3>Mean elevation per county</h3>
      <p class="prompt">"1) Clip the DEM to PA Boundaries and load the clipped DEM. 2) Generate the zonal statistics for every county in PA, focusing on average elevation. 3) Create a choropleth map … using a red color gradient."</p>
      <p class="tools">Tools: <code>gdal:cliprasterbymasklayer</code>, <code>native:zonalstatisticsfb</code>, thematic map · <a href="examples.html#tutorial-3-mean-elevation-per-county-choropleth">Tutorial</a></p>
    </div>
  </article>
  <article class="case">
    <figure><img src="assets/img/richland-terrain.webp" alt="Four DEM tiles merged and clipped to Richland County, with TPI, aspect, slope, hillshade and TRI outputs and the workflow graph." loading="lazy"><figcaption hidden>Richland County terrain analysis</figcaption></figure>
    <div class="case-body">
      <span class="badge l2">Intermediate · raster</span>
      <h3>Terrain characteristics</h3>
      <p class="prompt">"Merge the four DEMs into a single raster and perform terrain characteristic analysis for Richland County, including slope, aspect, hillshade, terrain ruggedness index (TRI), and topographic Position Index (TPI)."</p>
      <p class="tools">Tools: merge, clip by mask, slope, aspect, hillshade, TRI, TPI (7 algorithms)</p>
    </div>
  </article>
  <article class="case">
    <figure><img src="assets/img/obesity-risk-index.webp" alt="US counties shaded green by obesity risk behaviour index, with code and workflow graph." loading="lazy"><figcaption hidden>Obesity risk behaviour index</figcaption></figure>
    <div class="case-body">
      <span class="badge l2">Intermediate · vector + table</span>
      <h3>Obesity risk behaviour index</h3>
      <p class="prompt">"Generate an obesity risk behavior index of each county in the contiguous US by analyzing the rate of visits to unhealthy food retailers … and the visit rate to places that support physical activity … Visualize the results in a thematic map."</p>
      <p class="tools">Tools: <code>native:joinattributestable</code>, <code>qgis:advancedpythonfieldcalculator</code>, thematic map</p>
    </div>
  </article>
  <article class="case">
    <figure><img src="assets/img/affine-transform.webp" alt="Penn State building footprints before (grey) and after (brown) an affine transformation." loading="lazy"><figcaption hidden>Affine transformation of buildings</figcaption></figure>
    <div class="case-body">
      <span class="badge l2">Intermediate · vector</span>
      <h3>Affine transformation</h3>
      <p class="prompt">"Apply an affine transformation to the vector layer to scale by a factor of 2 and rotate 30 degrees."</p>
      <p class="tools">Tools: <code>native:affinetransform</code></p>
    </div>
  </article>
</div>

## Advanced

<div class="gallery">
  <article class="case">
    <figure><img src="assets/img/pa-fastfood-accessibility.webp" alt="PA fast food accessibility map in blue, scatter plot with regression line, and workflow graph." loading="lazy"><figcaption hidden>Fast food accessibility and obesity</figcaption></figure>
    <div class="case-body">
      <span class="badge l3">Advanced · vector + chart</span>
      <h3>Fast food accessibility vs. obesity</h3>
      <p class="prompt">"Could you analyze and visualize the fast food accessibility score for each county … using a thematic map with blue graduated colors. Then, analyze the correlation between the county-level obesity rate and the fast food accessibility score by drawing a scatter plot with a regression line."</p>
      <p class="tools">Tools: thematic map, <code>seaborn</code> · <a href="examples.html#tutorial-4-fast-food-accessibility-and-obesity">Tutorial</a></p>
    </div>
  </article>
  <article class="case">
    <figure><img src="assets/img/us-covid19.webp" alt="US counties shaded blue by COVID-19 cases, with code and workflow graph." loading="lazy"><figcaption hidden>COVID-19 cases by county</figcaption></figure>
    <div class="case-body">
      <span class="badge l3">Advanced · vector + table</span>
      <h3>COVID-19 cases by county</h3>
      <p class="prompt">"Could you show the spatial distribution of the COVID-19 cases across US counties?"</p>
      <p class="tools">Tools: thematic map. The agent joined a CSV to counties by FIPS on its own initiative.</p>
    </div>
  </article>
  <article class="case">
    <figure><img src="assets/img/akure-ndvi.webp" alt="NDVI raster of Akure clipped from Landsat imagery." loading="lazy"><figcaption hidden>NDVI of Akure</figcaption></figure>
    <div class="case-body">
      <span class="badge l3">Advanced · multiband raster</span>
      <h3>NDVI from Landsat</h3>
      <p class="prompt">"Generate the Normalized Difference Vegetation Index (NDVI) of Akure from these satellite imageries."</p>
      <p class="tools">Tools: <code>qgis:rastercalculator</code>, <code>native:reprojectlayer</code>, <code>gdal:cliprasterbymasklayer</code> · <a href="examples.html#tutorial-5-ndvi-from-landsat-imagery">Tutorial</a></p>
    </div>
  </article>
  <article class="case">
    <figure><img src="assets/img/residential-landuse.webp" alt="Cambridge residential land use in dark red, with total area of 10.446 square kilometres in the log." loading="lazy"><figcaption hidden>Residential area total</figcaption></figure>
    <div class="case-body">
      <span class="badge l3">Advanced · vector</span>
      <h3>Total residential area</h3>
      <p class="prompt">"Can you please select the residential area and calculate the total area covered by the residential area in square kilometers?"</p>
      <p class="tools">Tools: <code>native:extractbyattribute</code>, <code>native:fieldcalculator</code> · result 10.446 km². See <a href="examples.html#lesson-always-check-the-numbers">why the first run said 0</a>.</p>
    </div>
  </article>
  <article class="case">
    <figure><img src="assets/img/pa-dem-analysis.webp" alt="Pennsylvania DEM with a star-shaped workflow graph producing slope, aspect, hillshade, TRI, TPI and contour outputs." loading="lazy"><figcaption hidden>Pennsylvania DEM analysis</figcaption></figure>
    <div class="case-body">
      <span class="badge l3">Advanced · raster</span>
      <h3>Full DEM characterisation</h3>
      <p class="prompt">A Pennsylvania DEM analysed in one request into slope, aspect, hillshade, TRI, TPI, and contours. The graph fans out from a single input.</p>
      <p class="tools">Output: six derived layers</p>
    </div>
  </article>
  <article class="case">
    <figure><img src="assets/img/obesity-supermarket-correlation.webp" alt="US county points with the log showing a correlation table saved as HTML." loading="lazy"><figcaption hidden>Obesity and supermarket visits correlation</figcaption></figure>
    <div class="case-body">
      <span class="badge l3">Advanced · table + report</span>
      <h3>Correlation by state</h3>
      <p class="prompt">"Calculate the correlation coefficients between obesity rates and supermarket visit rates for each states in the USA. Let the result be in html format."</p>
      <p class="tools">Tools: <code>geopandas</code>, <code>pandas</code>. HTML table opened in Reports.</p>
    </div>
  </article>
</div>

## Explore further

- [Illustrated case studies on GitHub](https://github.com/Teakinboyewa/SpatialAnalysisAgent/blob/master/Case_Studies.md)
- [More demonstrations on the GIBD website](https://giscience.psu.edu/gis-copilot-demonstrations/)
- [Video demonstrations](videos.md)
