---
title: Interface and workflows
---

# Interface and workflows

[Documentation](index.md) / User guide

## Interface reference

| Tab | Purpose |
| --- | --- |
| **Request Page** | Choose data, describe a task, send a request, and read progress messages. |
| **Generated Code** | Inspect and edit Python; use **Run Code**, **Save as a Python File**, or **Open from File**. Read execution messages in **Output Window**. |
| **Geoprocessing Workflow** | Inspect the graph of operations and dependencies. Older screenshots call this Solution Graph. |
| **Reports** | View reports and visualizations when the task produces them. |
| **Settings** | Configure models, credentials, reasoning effort, code review, workspace, and tool documentation. |
| **Help** | Read the plugin's embedded guidance. |

These labels follow [the checked in interface](https://github.com/Teakinboyewa/SpatialAnalysisAgent/blob/master/SpatialAnalysisAgent_dockwidget_base.ui). Older screenshots may differ.

## Choose data

Check the layers to include in QGIS's Layers panel, or add datasets using **Load Data**. Confirm that input paths remain accessible. Name the intended layers and attribute fields in your request.

For distances and areas, specify units and an appropriate projected CRS. For raster calculations, identify bands, resolution, alignment, and NoData treatment where relevant.

## Write a clear request

Include inputs, operations, key parameters, and output requirements:

> Clip land_cover to study_boundary. Preserve raster categories and NoData values. Save the result as land_cover_clipped.tif in the workspace and add it to QGIS.

For longer tasks, explain the sequence and method. Define terms such as accessibility: straight line distance and network travel time require different analyses.

## Follow execution

The information panel displays data summaries, task breakdowns, selected tools, and execution status. A request can involve multiple model calls and debugging attempts.

**Interrupt** requests a stop. It does not undo code already executed or remove outputs already written. **Clear** clears interface content; manage saved files in the workspace.

The workflow graph describes the generated plan. Compare it with the final code and actual outputs, especially after debugging.

## Inspect and rerun code

Open **Generated Code** to inspect or edit the script. **Run Code** executes the editor contents inside QGIS. Before rerunning, check output paths: a repeated run may write to the same files or add layers again.

Use **Save as a Python File** to keep the script, and **Open from File** to load one into the editor. Saved scripts can depend on QGIS, installed packages, and the original data paths.

## Locate outputs

The configured workspace is supplied to code generation as the output location. Confirm actual paths in the code and execution messages. The workflow also produces graph artifacts, including GraphML and an HTML visualization. Not every analysis produces a report.

Preserve the request, input details, final script, and outputs together when you need a reproducible analysis.

[Continue: Configuration →](configuration.md)
