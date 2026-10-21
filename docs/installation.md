---
title: Installation
---

# Installation

[Documentation](index.md) / Installation

## Before you begin

You need QGIS desktop, its Python environment, a writable output directory, and access to a hosted model or a reachable local model server. Dependency installation requires internet access.

The repository's setup guidance requires **Python 3.11 or newer**. Although [metadata.txt](https://github.com/Teakinboyewa/SpatialAnalysisAgent/blob/master/metadata.txt) declares QGIS 3.0 as the minimum, that declaration is not a verified compatibility matrix for the current Python dependencies. Check the Python bundled with your QGIS installation when diagnosing compatibility.

## Install from the catalog

1. Open **Plugins → Manage and Install Plugins…** in QGIS.
2. Search for **AutonomousGIS-SpatialAnalysisAgent** and select **Install Plugin**.
3. Ensure the plugin is enabled.
4. Open **Spatial Analysis Agent** from the Plugins menu or its toolbar icon.

## Install from a ZIP

1. Download the [repository ZIP](https://github.com/Teakinboyewa/SpatialAnalysisAgent/archive/refs/heads/master.zip).
2. Open **Plugins → Manage and Install Plugins… → Install from ZIP**.
3. Choose the downloaded ZIP and select **Install Plugin**.
4. Enable and open the plugin.

Use this route when you specifically need the current repository version.

## Prepare dependencies

The plugin checks packages on startup and can prompt to install missing dependencies or resolve version mismatches. Allow installation to finish, then restart QGIS. If prompted to install pip, restart after that step too.

The dependency source is [install_packages/requirements.txt](https://github.com/Teakinboyewa/SpatialAnalysisAgent/blob/master/install_packages/requirements.txt). It includes NumPy, OpenAI, NetworkX, PyVis, LangChain packages, TOML parsers, GeoPandas, pandas, Pydantic, seaborn, and Rasterio.

**This file uses a custom `package:import_name` format. Do not pass it directly to `pip install -r`.** The plugin installer parses that format and applies the declared version constraints.

For manual repairs, identify the QGIS Python environment from **Plugins → Python Console**:

```python
import sys
print(sys.version)
print(sys.prefix)
```

Install packages into that QGIS environment. Installing into an unrelated system Python or virtual environment will not fix QGIS imports. Check compatibility before replacing QGIS supplied binary packages.

## Finish setup

Open **Settings**, configure your model, and choose a writable **Workspace Directory**. Continue to [Configuration](configuration.md) and [your first analysis](quickstart.md). For failures, see [Troubleshooting](troubleshooting.md).
