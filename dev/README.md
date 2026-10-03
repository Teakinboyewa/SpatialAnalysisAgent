# Debugging the plugin in QGIS from VS Code

QGIS runs the plugin straight from this repository; VS Code attaches to it, so
breakpoints, stepping and variable inspection work on the real plugin.

## One-time setup

1. **Install debugpy into QGIS's Python**, e.g. on Windows:
   ```
   "C:\Program Files\QGIS 3.x\apps\Python3xx\python.exe" -m pip install --user debugpy
   ```
2. **Add the debug block to your QGIS profile's `startup.py`**
   (`%APPDATA%\QGIS\QGIS3\profiles\default\python\startup.py` on Windows,
   `~/.local/share/QGIS/QGIS3/profiles/default/python/startup.py` on Linux,
   `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/startup.py` on macOS).
   It opens a debug port only when QGIS is started with `QGIS_DEBUGPY=1`:
   ```python
   import os, sys
   if os.environ.get("QGIS_DEBUGPY") == "1":
       try:
           import debugpy
           py = os.path.join(sys.prefix, "python.exe") if sys.platform == "win32" else os.path.join(sys.prefix, "bin", "python3")
           debugpy.configure(python=py)
           debugpy.listen(("127.0.0.1", int(os.environ.get("QGIS_DEBUGPY_PORT", "5678"))))
       except Exception as e:
           print(f"debugpy could not start: {e}")
   ```
3. **Let QGIS load the plugin from this repository** (with QGIS closed; remove any
   other copy of the plugin in the plugins folder first):
   ```
   Windows:  mklink /J "%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\SpatialAnalysisAgent-master" "<path to this repository>"
   Linux/macOS:  ln -s "<path to this repository>" "<profile>/python/plugins/SpatialAnalysisAgent-master"
   ```
   Then enable the plugin in *Plugins > Manage and Install Plugins*.
4. Install the **Plugin Reloader** plugin in QGIS.
5. In VS Code, install the **Python** and **Python Debugger** extensions.

## Everyday use

* Open this repository in VS Code, set breakpoints, run **Start QGIS and attach** (F5).
  It starts QGIS through `dev/qgis-debug` and attaches once QGIS is ready.
* If QGIS is already running in debug mode, use **Attach to QGIS**.
* After editing, reload the plugin with *Plugin Reloader* (or restart QGIS).

Qt background threads register with the debugger through `debug_support.py`.
Never commit API keys: settings the plugin writes must stay outside the repository.
