---
title: Development and custom tools
description: Repository layout, the analysis pipeline, adding tool documentation, and maintaining these docs.
---

# Development and custom tools

<p class="lead">For contributors: how the plugin is organised, how a request flows through the code, and how to teach GIS Copilot about new tools.</p>

## How a request flows through the code

```text
Send Request (SpatialAnalysisAgent_dockwidget.py)
  └─ worker thread runs SpatialAnalysisAgent_MyScript.py
       1. Data Eye           inspect layers → data overview      (SpatialAnalysisAgent_DataEye/)
       2. Task breakdown     model call                           (helper)
       3. Tool selection     match operations to tool IDs         (Tools_Documentation/*.toml)
       4. Workflow graph     NetworkX → GraphML + PyVis HTML      (graphs/)
       5. Code generation    prompt with tool docs + data overview
       6. Code review        optional model pass
       7. Execution          exec on the QGIS main thread         (SpatialAnalysisAgent_MainThread.py)
       8. Debugging          on error, regenerate and retry (up to 5 attempts)
  └─ dock widget streams progress, shows code, graph, and reports
```

Providers (OpenAI, GIBD, local OpenAI-compatible) are chosen in `SpatialAnalysisAgent_ModelProvider.py` based on the key and model selection.

## Repository map

| Path | Responsibility |
| --- | --- |
| `__init__.py` | QGIS `classFactory` entry point. |
| `SpatialAnalysisAgent.py` | Plugin lifecycle and QGIS integration. |
| `SpatialAnalysisAgent_dockwidget.py` | UI behavior, settings, workers, code execution, and feedback. |
| `SpatialAnalysisAgent_dockwidget_base.ui` | Qt Designer interface. |
| `SpatialAnalysisAgent/SpatialAnalysisAgent_MyScript.py` | Analysis orchestration. |
| `SpatialAnalysisAgent/SpatialAnalysisAgent_helper.py` | Prompts, model operations, and workflow helpers. |
| `SpatialAnalysisAgent/SpatialAnalysisAgent_ModelProvider.py` | Provider routing and configuration. |
| `SpatialAnalysisAgent/SpatialAnalysisAgent_MainThread.py` | Main thread coordination. |
| `SpatialAnalysisAgent/SpatialAnalysisAgent_DataEye/` | Input inspection. |
| `SpatialAnalysisAgent/Tools_Documentation/` | TOML tool knowledge and retrieval support. |
| `install_packages/` | Dependency checks and installation. |
| `dev/` | QGIS debugging setup. |
| `test/` | Existing plugin test scaffolding. |
| `Doc/` | Screenshots and research assets. |
| `docs/` | Current documentation pages. |

## Develop inside QGIS

The plugin depends on QGIS and Qt. Follow [dev/README.md](https://github.com/Teakinboyewa/SpatialAnalysisAgent/blob/master/dev/README.md) to load this checkout through a QGIS profile and attach VS Code with debugpy. Avoid loading duplicate copies. Restart QGIS or use Plugin Reloader after changes.

Keep QGIS object access and generated code execution consistent with existing main thread coordination. Background workers do not make arbitrary QGIS API calls thread safe.

## Tool documentation format

Adding documentation supplies model context; it does not install a processing provider or Python package. The described tool must be available in the target environment.

Each TOML file has five top level string fields:

```toml
tool_ID = 'native:buffer'
tool_name = 'Buffer'
brief_description = '''
Creates buffers around features. Use an appropriate projected CRS
for distances specified in meters.
'''
parameters = '''
INPUT: Input vector layer.
DISTANCE: Distance in the input layer coordinate units.
SEGMENTS: Segments used to approximate rounded corners.
DISSOLVE: Whether to merge overlapping buffers.
OUTPUT: Output destination.
'''
code_example = '''
import processing

result = processing.run('native:buffer', {
    'INPUT': input_layer,
    'DISTANCE': 500,
    'SEGMENTS': 5,
    'END_CAP_STYLE': 0,
    'JOIN_STYLE': 0,
    'MITER_LIMIT': 2,
    'DISSOLVE': False,
    'OUTPUT': output_path,
})
'''
```

This example assumes the caller defines `input_layer` and `output_path`. Test a complete runnable example against your QGIS version before contributing it. Inspect algorithm IDs and parameters in QGIS's Python Console:

```python
from qgis.core import QgsApplication
import processing

for algorithm in QgsApplication.processingRegistry().algorithms():
    print(algorithm.id(), algorithm.displayName())

processing.algorithmHelp('native:buffer')
```

For custom tools, use descriptive lowercase identifiers with underscores. Document packages, input assumptions, units, outputs, and failure conditions.

## Load a tool file

In **Settings**, choose **Local Machine** under the tool documentation controls. Select **QGIS Processing Tool** or **Customized Tool**, then select the TOML file. The importer copies it into the installed plugin and asks before replacing an existing filename.

The importer uses `QGIS_Tools` and `Customized_Tools`, while this repository also contains `Customized_tools`. Check retrieval paths on case sensitive filesystems before relying on a custom addition.

To contribute, add a tested file to the appropriate repository directory and submit a pull request. The [legacy contribution guide](https://github.com/Teakinboyewa/SpatialAnalysisAgent/blob/master/SpatialAnalysisAgent/Tools_Documentation/Tools_documentationREADME.MD) also describes the in-plugin GitHub dialog.

## Validate changes

- Exercise affected behavior inside QGIS with a small dataset.
- For analysis changes, check generated code, execution, saved outputs, and failure reporting.
- For provider changes, check routing and configuration without exposing credentials.
- Keep keys, `config.ini`, generated graphs, and workspace files out of commits.

The `test/` directory contains plugin scaffolding; it is not proof of end to end analysis correctness.

## Maintain documentation

These pages are Markdown with Jekyll front matter, readable on GitHub or in an editor. Start at [the documentation home](index.md). Match UI labels to the `.ui` file and behavior to the Python implementation. The older `help/source/` tree is Sphinx scaffolding and is not the source for these pages.

GitHub Pages publishes the `docs/` directory on `master`.

| File | Role |
| --- | --- |
| `_config.yml` | Site settings and the grouped sidebar navigation. Its order also drives the previous/next links. |
| `_layouts/default.html` | Page shell: top bar, sidebar, breadcrumbs, pager, footer, image lightbox. |
| `assets/style.css` | Theme, including dark mode, callouts, figures, gallery cards, and video embeds. |
| `assets/site.js` | Mobile menu, "On this page" contents, copy buttons, and image enlargement. |
| `assets/img/` | Web-optimised screenshots (WebP, max 1600 px wide) made from the originals in `Doc/`. |

Conventions:

- Add new pages to `navigation` in `_config.yml`.
- Link between pages with `.md` in Markdown links; GitHub Pages converts them. Links to files outside `docs/` must point to the GitHub repository.
- Use `<figure>` with descriptive `alt` text and a `<figcaption>` for screenshots. Add `class="narrow"` or `class="medium"` for small dialogs.
- Callouts use `<div class="callout note|tip|warning" markdown="1">` with a `<span class="callout-title">`.
- Keep screenshots free of API keys and personal paths. Replace an image when the UI changes, and say in the caption when a capture comes from an older release.

Preview locally with Jekyll:

```bash
cd docs
bundle exec jekyll serve   # or: gem install github-pages, then jekyll serve
```
