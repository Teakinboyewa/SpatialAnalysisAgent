---
title: Your first analysis
---

# Your first analysis

[Documentation](index.md) / Quickstart

Create a buffer around a point layer to learn the complete workflow.

## 1. Prepare the inputs

Load a point dataset into QGIS, such as health facilities. For this example, use a projected coordinate reference system with meter units. Rename the layer to `health_facilities` so the request can identify it clearly.

Check the layer in QGIS's Layers panel to include it in analysis. Uncheck unrelated layers. You can also add a dataset with the plugin's **Load Data** button.

## 2. Configure the plugin

In **Settings**, enter your hosted API key and select a model, or choose **local** and configure a server. Choose a writable **Workspace Directory** and leave **Review generated code** enabled for the first attempt. See [Configuration](configuration.md) for details.

## 3. Send a request

In **Request Page**, enter:

> Create a 500 meter buffer around every point in health_facilities. Keep overlapping buffers separate. Save the result as facility_buffers.gpkg in the workspace and add it to the QGIS project.

Select **Send Request**. The plugin generates and executes code as part of the request. Watch the information panel for data inspection, task breakdown, selected tools, execution, and debugging messages.

## 4. Check the result

1. Confirm that the output layer appears in QGIS and the file exists in the workspace.
2. Inspect the output CRS, geometry, and attributes. Measure a buffer to verify the intended distance.
3. Open **Generated Code** to inspect the script and **Geoprocessing Workflow** to review the operations.
4. Use **Save as a Python File** to preserve the script.

An execution success message confirms that code ran; it does not establish that the spatial method or result is correct.

## 5. Refine the analysis

For another distance, dissolved buffers, or a different output format, submit a new request specifying the full layer names and requirements. Check which layers are included before sending it.

[Continue: Interface and workflows →](user-guide.md)
