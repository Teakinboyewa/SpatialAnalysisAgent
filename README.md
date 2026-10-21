# GIS Copilot · Spatial Analysis Agent

Perform spatial analysis in QGIS using natural language. GIS Copilot examines your input layers, selects processing tools, generates Python code, and runs the workflow inside QGIS.

**[Read the documentation](https://teakinboyewa.github.io/SpatialAnalysisAgent/)** · [Install](docs/installation.md) · [First analysis](docs/quickstart.md) · [Report an issue](https://github.com/Teakinboyewa/SpatialAnalysisAgent/issues)

## Get started

1. In QGIS, open **Plugins → Manage and Install Plugins…**.
2. Search for **AutonomousGIS-SpatialAnalysisAgent** and install it.
3. Open the plugin from the Plugins menu or toolbar. Complete the dependency prompts, then restart QGIS.
4. In **Settings**, configure your API key and model, or select **local** and enter your model server details. Choose a writable workspace.
5. Load and check the layers to analyze, describe your task, and select **Send Request**.

See [Installation](docs/installation.md) for requirements and ZIP installation.

## What you can do

| Workflow | Examples |
| --- | --- |
| Vector analysis | Buffers, clipping, attribute queries, spatial joins |
| Raster analysis | Terrain analysis, raster clipping, zonal statistics |
| Reports and visualization | Charts, statistical summaries, thematic maps |
| Inspect and refine | View workflows, edit generated code, save Python scripts |
| Extend tool knowledge | Add TOML documentation for processing or custom tools |

Available operations depend on your data, QGIS providers, Python environment, and model. Inspect generated code and validate outputs before relying on an analysis.

## Documentation

- [Installation and dependencies](docs/installation.md)
- [Your first analysis](docs/quickstart.md)
- [Interface and workflows](docs/user-guide.md)
- [Models, settings, and data handling](docs/configuration.md)
- [Examples and case studies](docs/examples.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Development and custom tools](docs/development.md)

## Research and examples

The framework is described in *GIS Copilot: Towards an Autonomous GIS Agent for Spatial Analysis*, by Temitope Akinboyewa, Zhenlong Li, Huan Ning, and M. Naser Lessani. [Read the paper](https://doi.org/10.1080/17538947.2025.2497489).

Use the updated [case study PDF](Doc/Case%20Studies/GIS_Copilot_Case_Studies.pdf) and [test datasets](https://giscience.psu.edu/copilot_test/) in place of dataset links in the manuscript. Explore the [illustrated case studies](Case_Studies.md) and [QGIS plugin listing](https://plugins.qgis.org/plugins/SpatialAnalysisAgent-master/).

## Contribute

Start with the [developer guide](docs/development.md). Bug reports should include QGIS and Python versions, the selected model, reproduction steps, and relevant errors with credentials removed.

See [LICENSE](LICENSE) for licensing terms.
