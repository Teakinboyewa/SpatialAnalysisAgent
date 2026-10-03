#!/usr/bin/env bash
# Start QGIS with a debugpy port open, so VS Code can attach ("Attach to QGIS").
# Needs debugpy in QGIS's Python and the startup.py block described in dev/README.md.
# Set QGIS_BIN to override the QGIS executable.
export QGIS_DEBUGPY=1
export PYDEVD_DISABLE_FILE_VALIDATION=1
PORT="${QGIS_DEBUGPY_PORT:-5678}"
export QGIS_DEBUGPY_PORT="$PORT"

if [ -z "$QGIS_BIN" ]; then
  if [ -x "/Applications/QGIS.app/Contents/MacOS/QGIS" ]; then
    QGIS_BIN="/Applications/QGIS.app/Contents/MacOS/QGIS"
  elif [ -x "/Applications/QGIS-LTR.app/Contents/MacOS/QGIS" ]; then
    QGIS_BIN="/Applications/QGIS-LTR.app/Contents/MacOS/QGIS"
  else
    QGIS_BIN="$(command -v qgis)"
  fi
fi
if [ -z "$QGIS_BIN" ]; then echo "Could not find QGIS. Set QGIS_BIN."; exit 1; fi

echo "Starting QGIS for debugging: $QGIS_BIN"
nohup "$QGIS_BIN" >/dev/null 2>&1 &

# Wait (up to 3 minutes) until QGIS has opened the debug port.
for _ in $(seq 1 360); do
  if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then echo "QGIS is ready for debugging."; exit 0; fi
  sleep 0.5
done
echo "QGIS did not open the debug port. See dev/README.md."; exit 1
