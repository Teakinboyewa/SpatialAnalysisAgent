---
title: Configuration and data handling
---

# Configuration and data handling

[Documentation](index.md) / Configuration

## Hosted models

Enter your API key in **Settings** and select a model from **Models**. The current interface lists `gpt-6-luna`, `gpt-5.2`, `gpt-5.1`, `gpt-5`, `gpt-4o`, `gpt-4`, and `gpt-4o-mini`, plus `local`.

These are identifiers configured in this checkout, not a guarantee of availability through your provider. The initial UI selection is `gpt-6-luna`; saved settings can restore a different choice. If a request returns a model error, select an identifier supported by your endpoint and account.

Keys containing `gibd-services` route to the GIBD service. Other hosted keys use the plugin's OpenAI provider path. See the [GIBD website](https://www.gibd.online/) for account details. Historical quota offers in earlier guides should not be treated as current entitlements.

## Local models

1. Start your model server and load a model.
2. Choose **local** under **Models**.
3. Enter its OpenAI compatible **Base URL**, including `/v1`, for example `http://localhost:11434/v1`.
4. Select **Fetch models**, then choose an exact model identifier. You can also type the identifier.
5. Send a small request to verify generation and execution.

Discovery probes OpenAI compatible, Ollama, and LM Studio listing endpoints. Successful discovery does not guarantee that generation will work for every returned identifier.

The UI stores the URL and model under `[LocalModel]` in the configuration and writes `no-api` as the API key placeholder. This path is intended for servers that accept that placeholder; it does not expose a custom authentication field. Set the Base URL explicitly. Hardware requirements depend on your model and server.

## Analysis settings

| Setting | Effect |
| --- | --- |
| **Workspace Directory** | Supplies an output directory to code generation. Select **Change** to choose a writable folder. |
| **Review generated code** | Adds model review before execution. It increases processing time and does not guarantee correctness. |
| **Reasoning Effort** | Sets effort for supported model paths. Values and defaults depend on the selected model. |

Some stages can use a different model through `get_model_for_operation` in [SpatialAnalysisAgent_helper.py](https://github.com/Teakinboyewa/SpatialAnalysisAgent/blob/master/SpatialAnalysisAgent/SpatialAnalysisAgent_helper.py). Do not assume every API call uses the dropdown selection unchanged.

Older guides mention Chat Mode and a RAG toggle. The current checked in interface does not expose those controls, so they are not setup steps for this version.

## Credentials and request data

The plugin writes credentials to `SpatialAnalysisAgent/config.ini` within the installed plugin and also uses QGIS settings for some values. Local configuration is not an encrypted credential vault. Exclude keys and configuration files from commits, screenshots, and issue reports.

Local key storage does not mean analysis requests stay local. The selected endpoint receives prompts assembled from the request, data overviews, tool documentation, code, and error context as the workflow requires. Check the destination and prompt contents before using sensitive data. A model server on another machine receives requests on that machine even when the UI option says **local**.

The GIBD path also supports feedback submission. Repository service guidance describes logging queries, workflow details, generated code, errors, and limited dataset metadata. Consult the service for current terms. These pages do not establish provider retention policies or promise that prompts contain no dataset content.

[Continue: Troubleshooting →](troubleshooting.md)
