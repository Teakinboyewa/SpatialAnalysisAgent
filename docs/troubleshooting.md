---
title: Troubleshooting
---

# Troubleshooting

[Documentation](index.md) / Troubleshooting

## Plugin does not load

Confirm it is enabled in **Plugins → Manage and Install Plugins…**. Restart QGIS after dependency changes. Inspect QGIS log messages or the Python Console for the first failing import.

If using a development checkout, confirm which plugin copy QGIS loads. Avoid loading a catalog installation and a development copy together.

## Missing modules or package incompatibilities

Allow the dependency check to finish. If installation fails, record the package and error, then check network access, permissions, and the Python environment belonging to QGIS.

Use [install_packages/requirements.txt](https://github.com/Teakinboyewa/SpatialAnalysisAgent/blob/master/install_packages/requirements.txt) as the version reference. Its custom format is not compatible with a direct `pip install -r` command. Installing into an unrelated Python environment will not repair QGIS imports.

For NumPy, pandas, GeoPandas, and Rasterio errors, inspect the full traceback before changing versions: they can depend on binary components bundled with QGIS.

## Authentication or unavailable model

Check the key in **Settings**, account access, and the selected model identifier. A model listed in the interface may be unavailable through your endpoint. For GIBD, verify that the key belongs to that service. Never include keys in a bug report.

## Local model connection or discovery fails

1. Confirm the server is running and a model is loaded.
2. Check the Base URL, including host, port, and `/v1`.
3. Use **Fetch models** and inspect any endpoint error.
4. Choose the exact callable model identifier rather than a display label.
5. If discovery works but generation fails, inspect the server's generation error and authentication requirements.

`localhost` refers to the computer running QGIS. A server elsewhere needs a reachable address. The current local UI saves the `no-api` authentication placeholder; see [Configuration](configuration.md).

## Request seems slow

Analysis involves several model calls, optional code review, geoprocessing, and possible debugging. Check progress before submitting another request. Try a smaller dataset or one operation to isolate the slow stage.

**Interrupt** requests a stop but cannot undo completed work. Inspect partial outputs before rerunning.

## Execution succeeds but the result is wrong

Inspect the final code, input layers, field names, CRS, units, raster bands, NoData handling, and output paths. Compare a small sample against a manually run QGIS processing operation. Rewrite the request with missing assumptions made explicit.

## No report or graph appears

Not every task produces a report. Check whether execution reached the relevant stage and inspect messages for file paths or errors. Compare a generated graph with the final code; it describes a plan, not a verified execution record.

## Report an issue

Include the following in a [GitHub issue](https://github.com/Teakinboyewa/SpatialAnalysisAgent/issues):

- QGIS, Python, operating system, and plugin versions.
- Installation method and selected provider/model.
- Minimal request, shareable input schema or sample, and reproduction steps.
- Expected behavior, actual behavior, and relevant traceback.

Remove credentials and sensitive paths, prompts, or dataset contents before posting.
