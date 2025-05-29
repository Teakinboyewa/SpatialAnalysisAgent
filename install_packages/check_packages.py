import os
import platform
import sys
import importlib
import subprocess
from qgis.PyQt.QtWidgets import QMessageBox, QProgressDialog, QApplication
from qgis.PyQt.QtCore import QSettings, QThread, pyqtSignal, Qt
from concurrent.futures import ThreadPoolExecutor
# from pip._internal import main as pip_main

# class LibraryInstallThread(QThread):
#     """
#     Worker thread that installs one package at a time
#     and emits progress so the GUI can update.
#     """
#     progress = pyqtSignal(int, str)   # (index, package_name)
#     finished = pyqtSignal(bool)       # True if all succeeded
#
#     def __init__(self, libraries):
#         super().__init__()
#         self.libraries = libraries
#
#     def run(self):
#         success = True
#         for i, pkg in enumerate(self.libraries):
#             self.progress.emit(i, pkg)
#             # install in‐process
#             ret = subprocess.call([sys.executable, "-m", "pip", "install", pkg])
#             if ret != 0:
#                 success = False
#                 # optionally continue to next or break()
#         self.finished.emit(success)


# def install_libraries_threaded(libraries, parent_widget=None):
#     """
#     Pop up a QProgressDialog and install packages in a background thread.
#     """
#     dg = QProgressDialog("Installing dependencies…", "Abort", 0, len(libraries), parent_widget)
#     dg.setWindowModality(Qt.WindowModal)
#     dg.setMinimumDuration(0)
#     dg.setAutoClose(True)
#     dg.show()
#
#     worker = LibraryInstallThread(libraries)
#
#     # Update dialog on each package
#     def on_progress(idx, pkg):
#         dg.setValue(idx)
#         dg.setLabelText(f"Installing {pkg} ({idx+1}/{len(libraries)})")
#         QApplication.processEvents()
#     worker.progress.connect(on_progress)
#
#     # When done, close dialog & notify
#     def on_finished(ok):
#         dg.setValue(len(libraries))
#         dg.close()
#         if ok:
#             QMessageBox.information(parent_widget, "Done",
#                 "All packages installed successfully.\nPlease restart QGIS.")
#         else:
#             QMessageBox.warning(parent_widget, "Partial Failure",
#                 "Some packages failed to install.\nCheck the log for details.")
#     worker.finished.connect(on_finished)
#
#     # Allow user to cancel
#     dg.canceled.connect(worker.terminate)
#
#     worker.start()


# def check_and_install_libraries(filename):
#     """Check and install required third-party Python libraries."""
#     settings = QSettings()
#     required_libraries = read_libraries_from_file(filename)
#     cached_libraries = settings.value("cached_libraries", [])
#
#     if not cached_libraries or cached_libraries != required_libraries:
#         missing_packages = check_missing_libraries(required_libraries)
#
#     if missing_packages:
#         message = "The following Python packages are required to use the plugin:\n\n"
#         message += "\n".join(missing_packages)
#         message += "\n\nWould you like to install them now? After installation, please restart QGIS."
#
#         # Display the message box to the user
#         reply = QMessageBox.question(None, 'Missing Dependencies', message,
#                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
#         if reply == QMessageBox.Yes:
#             # Install the missing packages
#             # install_libraries(missing_packages)
#             install_libraries_threaded(missing_packages, parent_widget=None)
#
#             # settings.setValue("libraries_installed", True)  # Mark libraries as installed
#             settings.setValue("cached_libraries", required_libraries)  # Cache the list of installed libraries
#
#         elif reply == QMessageBox.No:
#             # Close the current dialog or window when the user clicks "No"
#             return  # Stop installation if user declines
#     else:
#         # settings.setValue("libraries_installed", True)  # Mark libraries as installed if none are missing
#         settings.setValue("cached_libraries", required_libraries)


def check_and_install_libraries(filename, parent_widget=None):
    settings = QSettings()
    required_libraries = read_libraries_from_file(filename)
    cached_libraries = settings.value("cached_libraries", [])

    if not cached_libraries or cached_libraries != required_libraries:
        missing_packages = check_missing_libraries(required_libraries)

        if missing_packages:
            # package_names = [pkg for pkg, _ in missing_packages]
            package_names = missing_packages
            # pip_command = f"pip install {' '.join(package_names)}"

            reply = QMessageBox.question(parent_widget, 'Missing Dependencies',
                                         "The following Python packages are required:\n\n"
                                         + "\n".join(package_names) +
                                         "\n\nWould you like to install them now?\n\n"
                                         "This will open a terminal window with the install command.",
                                         QMessageBox.Yes | QMessageBox.No)

            if reply == QMessageBox.Yes:
                install_libraries(package_names)

                # Cache the libraries to avoid checking again
                settings.setValue("cached_libraries", required_libraries)


def install_libraries(libraries):
    """Install missing libraries using pip."""
    import subprocess, platform, os, sys
    # Get the os type
    system = platform.system()
    # Grab the “real” interpreter path for each OS
    if system == "Windows":
        # usually C:\…\apps\PythonXX\python.exe
        python_exe = getattr(sys, "_base_executable", None)
        if not python_exe or "qgis" in os.path.basename(python_exe).lower():
            python_exe = os.path.join(sys.prefix, "python.exe")


    elif system == "Linux":
        # usually /usr/bin/python3 or under sys.prefix/bin/
        python_exe = getattr(sys, "_base_executable", None)
        if not python_exe or "qgis" in os.path.basename(python_exe).lower():
            candidate = os.path.join(sys.prefix, "bin", "python3")
            python_exe = candidate if os.path.isfile(candidate) else "python3"

    elif system == "Darwin":
        # macOS QGIS bundles are similar to Linux
        python_exe = getattr(sys, "_base_executable", None)
        if not python_exe or "qgis" in os.path.basename(python_exe).lower():
            candidate = os.path.join(sys.prefix, "bin", "python3")
            python_exe = candidate if os.path.isfile(candidate) else "python3"

    else:
        raise RuntimeError(f"Unsupported OS: {system!r}")


    cmd = [python_exe, "-m", "pip", "install", "--user"] + libraries
    subprocess.check_call(cmd)


    # for pkg in libraries:
    #     # 4. Install packages (for example, geopandas)
    #     subprocess.check_call([
    #         python_exe, "-m", "pip", "install", "--user", pkg
    #     ])

# def launch_terminal_with_command(command, parent_widget=None):
#     """
#     Launch a new terminal window and run a command, keeping the terminal open.
#     Works on Windows, macOS, and Linux.
#     """
#     import subprocess, platform, os, sys
#     from qgis.PyQt.QtWidgets import QMessageBox
#
#
#
#     # Get QGIS Python path
#     qgis_python = sys.executable  # This is the Python running inside QGIS
#
#     # Modify command to use QGIS's Python
#     full_command = f'"{qgis_python}" -m {command}' if command.startswith("pip") else command
#
#
#     try:
#         if platform.system() == "Windows":
#             subprocess.Popen(
#                 ['start', 'cmd', '/k', full_command],
#                 shell=True,
#                 creationflags=subprocess.CREATE_NEW_CONSOLE
#             )
#
#         elif platform.system() == "Darwin":  # macOS
#             subprocess.Popen(['osascript', '-e',
#                               f'tell application "Terminal" to do script "{full_command}"'])
#
#         elif platform.system() == "Linux":
#             # Tries common terminal emulators — adapt if needed
#             for terminal in ['x-terminal-emulator', 'gnome-terminal', 'konsole', 'xfce4-terminal']:
#                 try:
#                     subprocess.Popen([terminal, '-e', full_command])
#                     break
#                 except FileNotFoundError:
#                     continue
#             else:
#                 raise EnvironmentError("No supported terminal emulator found.")
#         else:
#             raise EnvironmentError("Unsupported operating system.")
#     except Exception as e:
#         QMessageBox.critical(parent_widget, "Error Opening Terminal",
#                              f"Failed to open terminal automatically.\n\n"
#                              f"Please run this command manually:\n\n{full_command}\n\nError:\n{str(e)}")




def check_missing_libraries(libraries):
        """Function to install missing libraries using pip."""
        missing_packages = []

        # Use ThreadPoolExecutor for parallel checking
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
        return (library, True)  # Library is missing


# def install_libraries(libraries):
#     """Install missing libraries using pip."""
#
#     for pkg in libraries:
#         ret = pip_main(["install", pkg])
#         if ret != 0:
#             print(f"Error installing {pkg} (exit code {ret})")
#         else:
#             print(f"Installed {pkg}")


def read_libraries_from_file(filename):
    """Read the list of libraries and their import paths from a text file."""
    libraries = []
    with open(filename, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                # Each line is in the format: library_name:module_name
                library, module = line.strip().split(':')
                libraries.append((library, module))
    return libraries








