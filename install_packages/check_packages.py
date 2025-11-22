import os
import platform
import sys
import importlib
import subprocess
import pkg_resources
from qgis.PyQt.QtWidgets import QMessageBox, QProgressDialog, QApplication
from qgis.PyQt.QtCore import QSettings, QThread, pyqtSignal, Qt
from concurrent.futures import ThreadPoolExecutor




def check_missing_libraries(libraries):
        """Function to install missing libraries using pip."""
        missing_packages = []
        with ThreadPoolExecutor() as executor:
            results = executor.map(check_library, libraries)

        for library, missing in results:
            if missing:
                missing_packages.append(library)
        return missing_packages


def check_library(library_info):
   """Check if a library is installed, return (library, is_missing)."""
   library, module = library_info
   try:
       importlib.import_module(module)
       return (library, False)  # Library is installed
   except ImportError:

       installed = {pkg.key.lower().replace("-", "_") for pkg in pkg_resources.working_set}
       normalized_name = library.lower().replace("-", "_")
       if normalized_name in installed:
           return (library, False)  # Installed but not importable
       return (library, True)  # Not installed




def check_library_installed_only(distribution_name):
    """
    Check if a distribution (installed via pip) exists in the environment.
    This does not check if the module is importable.
    Returns (distribution_name, is_missing: bool)
    """
    installed = {pkg.key for pkg in pkg_resources.working_set}
    normalized_name = distribution_name.lower().replace("-", "_")

    if normalized_name in installed:
        return (distribution_name, False)
    else:
        return (distribution_name, True)


def read_libraries_from_file(filename):
    """Read the list of libraries and their import paths from a text file."""
    libraries = []
    with open(filename, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                # Each line is in the format: library_name:module_name
                library, module = line.strip().split(':')
                libraries.append((library.strip(), module.strip()))
    return libraries