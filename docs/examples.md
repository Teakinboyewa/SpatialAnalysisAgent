---
title: Examples and case studies
---

# Examples and case studies

[Documentation](index.md) / Examples

Replace these layer and field names with your own. Check the inputs in the Layers panel and validate each output.

## Attribute selection

> From counties, select features where population is greater than 50000. Save the selected features as populous_counties.gpkg in the workspace and add the output to QGIS.

Verify the field name and type. Specify whether you want a selection or a separate dataset.

## Raster clipping

> Clip elevation to study_boundary. Preserve the input resolution and NoData handling. Save the result as elevation_clipped.tif in the workspace and add it to QGIS.

Inspect the extent, CRS, pixel size, and boundary coverage afterward.

## Zonal statistics

> Calculate mean elevation from elevation for each polygon in counties. Save a new GeoPackage layer with the county identifiers and mean elevation. Create a histogram of the county means.

Check raster band selection, NoData treatment, and polygons without valid raster cells.

## Multiple operations

> Clip elevation to study_boundary, calculate slope in degrees, and summarize mean slope for each polygon in management_zones. Save the clipped elevation, slope raster, and zone summary in the workspace. Add the spatial outputs to QGIS.

Check intermediate results before interpreting the summary. The slope method, elevation units, and CRS affect the result.

## Research resources

- [Illustrated case studies](https://github.com/Teakinboyewa/SpatialAnalysisAgent/blob/master/Case_Studies.md): basic, intermediate, and advanced workflows with output images.
- [Updated case study PDF](https://github.com/Teakinboyewa/SpatialAnalysisAgent/blob/master/Doc/Case%20Studies/GIS_Copilot_Case_Studies.pdf).
- [Test cases and datasets](https://giscience.psu.edu/copilot_test/).
- [Additional demonstrations](https://giscience.psu.edu/gis-copilot-demonstrations/).

Use the updated dataset references in place of older manuscript links. Case studies show specific runs; different models, providers, and QGIS environments can produce different results.
