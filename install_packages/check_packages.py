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




# ***********************************
def parse_requirements_with_versions(filename):
    """
    Parse requirements file with version specifications.
    Returns a dict: {package_name: version_spec} where version_spec is the full spec (e.g., '==2.3.5')
    or None if no version is specified.
    """
    requirements = {}
    print(f"[PARSE DEBUG] Parsing requirements from {filename}")
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            # Format: package_name:module_name or package_name==version:module_name
            if ':' not in line:
                print(f"[PARSE DEBUG] Skipping line (no colon): {line}")
                continue

            package_spec = line.split(':')[0].strip()

            # Extract package name and version specifier
            for op in ['==', '>=', '<=', '!=', '>', '<']:
                if op in package_spec:
                    parts = package_spec.split(op, 1)
                    package_name = parts[0].strip()
                    version_spec = op + parts[1].strip()
                    requirements[package_name] = version_spec
                    print(f"[PARSE DEBUG] Found: {package_name} -> {version_spec}")
                    break
            else:
                # No version specified
                requirements[package_spec] = None
                print(f"[PARSE DEBUG] Found (no version): {package_spec}")

    print(f"[PARSE DEBUG] Total requirements parsed: {requirements}")
    return requirements


def get_installed_version(package_name):
    """
    Get the installed version of a package.
    Returns version string or None if not installed.
    Tries multiple methods to handle different package name formats.
    """
    # Method 1: Try importlib.metadata (most reliable)
    try:
        return version(package_name)
    except:
        pass

    # Method 2: Try pkg_resources with different name normalizations
    try:
        dist = pkg_resources.get_distribution(package_name)
        return dist.version
    except:
        pass

    # Method 3: Try with underscores and hyphens swapped
    try:
        normalized = package_name.replace('-', '_')
        return version(normalized)
    except:
        pass

    try:
        normalized = package_name.replace('_', '-')
        return version(normalized)
    except:
        pass

    # Method 4: Try all installed packages (fallback)
    try:
        installed = {pkg.key for pkg in pkg_resources.working_set}
        normalized_name = package_name.lower().replace("-", "_")
        for pkg_key in installed:
            if pkg_key.lower().replace("-", "_") == normalized_name:
                dist = pkg_resources.get_distribution(pkg_key)
                return dist.version
    except:
        pass

    return None


def check_version_mismatches(requirements_file):
    """
    Check which packages have version mismatches between requirements and installed.
    Returns a dict: {package_name: (required_spec, installed_version)}
    """
    requirements = parse_requirements_with_versions(requirements_file)
    mismatches = {}

    print(f"[VERSION CHECK DEBUG] Checking {len(requirements)} packages from {requirements_file}")

    for package_name, required_spec in requirements.items():
        installed_version = get_installed_version(package_name)

        print(f"[VERSION CHECK DEBUG] {package_name}: required={required_spec}, installed={installed_version}")

        if required_spec is None:
            # No version requirement, skip if installed
            if installed_version is None:
                mismatches[package_name] = (None, None)
        else:
            # Check if version matches
            if installed_version is None:
                # Package not installed
                print(f"  -> NOT INSTALLED (mismatch)")
                mismatches[package_name] = (required_spec, None)
            else:
                # Check if versions match
                satisfies = version_satisfies(installed_version, required_spec)
                print(f"  -> version_satisfies={satisfies}")
                if not satisfies:
                    print(f"  -> MISMATCH DETECTED")
                    mismatches[package_name] = (required_spec, installed_version)
                else:
                    print(f"  -> OK (versions match)")

    print(f"[VERSION CHECK DEBUG] Found {len(mismatches)} mismatches: {mismatches}")
    return mismatches


def version_satisfies(installed, required_spec):
    """
    Check if installed version satisfies the required specification.
    """
    if '==' in required_spec:
        required = required_spec.split('==')[1].strip()
        return installed == required
    elif '>=' in required_spec:
        required = required_spec.split('>=')[1].strip()
        return compare_versions(installed, required) >= 0
    elif '<=' in required_spec:
        required = required_spec.split('<=')[1].strip()
        return compare_versions(installed, required) <= 0
    elif '!=' in required_spec:
        required = required_spec.split('!=')[1].strip()
        return installed != required
    elif '>' in required_spec:
        required = required_spec.split('>')[1].strip()
        return compare_versions(installed, required) > 0
    elif '<' in required_spec:
        required = required_spec.split('<')[1].strip()
        return compare_versions(installed, required) < 0
    return True


def compare_versions(v1, v2):
    """
    Compare two version strings.
    Returns: -1 if v1 < v2, 0 if equal, 1 if v1 > v2
    """
    try:
        return (pkg_resources.parse_version(v1) > pkg_resources.parse_version(v2)) - \
               (pkg_resources.parse_version(v1) < pkg_resources.parse_version(v2))
    except:
        return 0


def install_specific_versions(requirements_file, show_progress=True):
    """
    Install packages with the exact versions specified in requirements file.
    Handles both packages with and without version specifications.

    Args:
        requirements_file: Path to requirements file
        show_progress: Show progress dialog during installation

    Returns:
        tuple: (success: bool, message: str, installed_count: int)
    """
    try:
        requirements = parse_requirements_with_versions(requirements_file)

        if not requirements:
            return False, "No packages found in requirements file.", 0

        # Build list of packages with versions
        packages_to_install = []
        for package_name, version_spec in requirements.items():
            if version_spec:
                packages_to_install.append(package_name + version_spec)
            else:
                packages_to_install.append(package_name)

        print(f"Installing packages: {packages_to_install}")

        # Show progress dialog if requested
        progress = None
        if show_progress:
            progress = QProgressDialog("Installing packages...", None, 0, 0)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

        try:
            subprocess.check_call(['python3', '-m', 'pip', 'install'] + packages_to_install)
        finally:
            if progress:
                progress.close()

        return True, f"Successfully installed {len(packages_to_install)} packages.", len(packages_to_install)

    except subprocess.CalledProcessError as e:
        return False, f"Installation failed: {str(e)}", 0
    except FileNotFoundError as e:
        return False, f"Requirements file not found: {str(e)}", 0
    except Exception as e:
        return False, f"Error during installation: {str(e)}", 0


def check_and_install_with_versions(requirements_file):
    """
    Check if packages are installed with correct versions.
    If there are mismatches or missing packages, prompt user to install them.

    Args:
        requirements_file: Path to requirements file with version specifications
    """
    try:
        # Check for version mismatches
        mismatches = check_version_mismatches(requirements_file)

        if mismatches:
            # Build detailed message showing what's wrong
            message = "The following packages have version mismatches:\n\n"
            packages_to_fix = []

            for package_name, (required_spec, installed_version) in mismatches.items():
                if installed_version is None:
                    message += f"• {package_name}: NOT INSTALLED (required: {required_spec})\n"
                    packages_to_fix.append(package_name + required_spec)
                else:
                    message += f"• {package_name}: installed {installed_version}, required {required_spec}\n"
                    packages_to_fix.append(package_name + required_spec)

            message += "\nWould you like to install the correct versions now? After installation, please restart QGIS."

            # Display message and ask user
            reply = QMessageBox.question(None, 'Version Mismatch', message,
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                # Install with correct versions
                print(f"Installing packages with correct versions: {packages_to_fix}")
                subprocess.check_call(['python3', '-m', 'pip', 'install', '--force-reinstall'] + packages_to_fix)
                QMessageBox.information(None, 'Installation Complete',
                                      f'Successfully installed {len(packages_to_fix)} packages with correct versions.\n\nPlease restart QGIS.')

        else:
            print("All packages are installed with correct versions!")

    except Exception as e:
        print(f"Error checking versions: {e}")
        QMessageBox.warning(None, 'Error', f'Error checking package versions: {str(e)}')
