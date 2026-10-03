# -*- coding: utf-8 -*-
"""Development helper for debugging the plugin from VS Code (see dev/README.md).

debugpy only follows threads started with Python's ``threading`` module. The
plugin's background work runs in Qt threads (QThread), so each of their
``run()`` methods calls ``debug_this_thread()`` to make breakpoints there
work. It does nothing unless debugpy was started (by the QGIS startup.py
block), so normal use is unaffected.

Usage, as the first line of every QThread.run() that runs plugin code:

    from .debug_support import debug_this_thread
    ...
    def run(self):
        debug_this_thread()
        ...
"""
import sys


def debug_this_thread():
    debugpy = sys.modules.get("debugpy")
    if debugpy is None:
        return
    try:
        debugpy.debug_this_thread()
    except Exception:
        pass
