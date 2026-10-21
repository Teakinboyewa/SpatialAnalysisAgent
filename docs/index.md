---
title: GIS Copilot documentation
---

# GIS Copilot documentation

Use natural language to perform spatial analysis inside QGIS. GIS Copilot inspects input layers, selects tools, builds a workflow, generates Python code, and executes it in QGIS.

## Find your next step

| Goal | Guide |
| --- | --- |
| Install the plugin and dependencies | [Installation](installation.md) |
| Complete a first analysis | [Quickstart](quickstart.md) |
| Understand tabs, execution, and outputs | [User guide](user-guide.md) |
| Configure credentials or a local model | [Configuration](configuration.md) |
| Explore prompts and research datasets | [Examples](examples.md) |
| Resolve errors | [Troubleshooting](troubleshooting.md) |
| Develop the plugin or add tool knowledge | [Development](development.md) |

## How analysis works

1. Choose your input layers, describe the task, and select a workspace.
2. The plugin inspects the data and develops a task breakdown.
3. It selects tools, retrieves documentation, and generates a workflow graph.
4. It generates Python, optionally reviews it, and executes it in QGIS.
5. It attempts to debug execution errors and presents code, layers, and any reports produced.

Analysis requests execute generated code automatically. The Generated Code tab lets you inspect and edit it afterward; it is not an approval gate before initial execution. Validate the spatial method and outputs before relying on a result.

These pages describe the version 1.4 repository interface. Catalog releases and older screenshots may differ. See [plugin metadata](https://github.com/Teakinboyewa/SpatialAnalysisAgent/blob/master/metadata.txt) for release history and [the repository overview](https://github.com/Teakinboyewa/SpatialAnalysisAgent/blob/master/README.md) for project and research links.

[Get started →](installation.md)
