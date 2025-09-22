"""
Smart Debugging System for SpatialAnalysisAgent
Provides intelligent error analysis, pattern recognition, and adaptive learning for debugging spatial analysis code.
"""

import re
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class ErrorPatternMatcher:
    """Handles error pattern recognition and categorization"""

    def __init__(self):
        self.error_patterns = {
            "import_errors": {
                "patterns": [
                    r"ImportError", r"ModuleNotFoundError", r"No module named",
                    r"cannot import name", r"DLL load failed"
                ],
                "solutions": [
                    "Check if the required library is installed",
                    "Use alternative import statements",
                    "Import from correct module path",
                    "Try importing from PyQt5 instead of qgis.PyQt",
                    "Restart QGIS if DLL issues persist"
                ],
                "severity": "high",
                "category": "environment"
            },
            "qgis_specific": {
                "patterns": [
                    r"QgsVectorLayer.*not found", r"processing\.run.*error",
                    r"AttributeError.*Qgs", r"QgsProject.*instance",
                    r"QgsVectorFileWriter.*error", r"QgsApplication.*not initialized"
                ],
                "solutions": [
                    "Ensure QGIS environment is properly initialized",
                    "Import QGIS classes from correct modules (qgis.core)",
                    "Use processing.run with correct algorithm ID",
                    "Check if layer is valid before processing",
                    "Verify QgsApplication is running"
                ],
                "severity": "high",
                "category": "qgis_environment"
            },
            "data_path_errors": {
                "patterns": [
                    r"FileNotFoundError", r"No such file", r"Invalid data source",
                    r"cannot open.*shp", r"Path does not exist", r"Permission denied"
                ],
                "solutions": [
                    "Verify data file paths exist and are accessible",
                    "Check file permissions and access rights",
                    "Use forward slashes or raw strings for Windows paths",
                    "Ensure data files are not corrupted or locked",
                    "Check if file extension matches actual format"
                ],
                "severity": "medium",
                "category": "data_access"
            },
            "processing_algorithm_errors": {
                "patterns": [
                    r"Algorithm.*not found", r"native:.*not available", r"gdal:.*error",
                    r"Parameter.*required", r"Invalid algorithm", r"Algorithm.*does not exist"
                ],
                "solutions": [
                    "Check algorithm ID spelling and availability",
                    "Use correct parameter names for algorithm",
                    "Ensure all required parameters are provided",
                    "Try alternative algorithm if available",
                    "Check QGIS processing provider is enabled"
                ],
                "severity": "high",
                "category": "algorithm"
            },
            "geometry_errors": {
                "patterns": [
                    r"Geometry.*invalid", r"TopologyException", r"GEOS.*error",
                    r"Self-intersection", r"Invalid polygon", r"Ring.*not closed"
                ],
                "solutions": [
                    "Use buffer(0) to fix minor geometry issues",
                    "Apply geometry validation before processing",
                    "Use native:fixgeometries algorithm",
                    "Check for and remove self-intersecting polygons",
                    "Ensure polygon rings are properly closed"
                ],
                "severity": "medium",
                "category": "geometry"
            },
            "field_errors": {
                "patterns": [
                    r"Field.*not found", r"AttributeError.*field", r"Column.*does not exist",
                    r"Field name.*too long", r"Invalid field name"
                ],
                "solutions": [
                    "Check field names in the attribute table",
                    "Truncate field names to 10 characters for shapefiles",
                    "Use layer.fields() to list available fields",
                    "Ensure field exists before accessing",
                    "Use valid field name characters (no spaces/special chars)"
                ],
                "severity": "medium",
                "category": "attributes"
            },
            "memory_errors": {
                "patterns": [
                    r"MemoryError", r"Out of memory", r"Cannot allocate memory",
                    r"Killed.*memory"
                ],
                "solutions": [
                    "Process data in smaller chunks or tiles",
                    "Use temporary files instead of memory layers",
                    "Simplify geometries before processing",
                    "Close unused layers to free memory",
                    "Increase system virtual memory"
                ],
                "severity": "high",
                "category": "performance"
            },
            "coordinate_system_errors": {
                "patterns": [
                    r"CRS.*not found", r"Projection.*error", r"Transform.*failed",
                    r"EPSG.*invalid", r"Coordinate.*out of range"
                ],
                "solutions": [
                    "Check coordinate reference system is valid",
                    "Reproject layers to matching CRS before operations",
                    "Use EPSG codes for standard projections",
                    "Verify coordinate values are within valid range",
                    "Set project CRS to match data CRS"
                ],
                "severity": "medium",
                "category": "projection"
            }
        }

    def match_error_pattern(self, error_msg: str) -> Tuple[Optional[str], Dict]:
        """Match error message to known patterns"""
        for category, info in self.error_patterns.items():
            for pattern in info["patterns"]:
                if re.search(pattern, error_msg, re.IGNORECASE):
                    return category, info
        return None, {}


class ContextAnalyzer:
    """Analyzes error context for more targeted debugging"""

    def __init__(self):
        self.operation_contexts = {
            "join": {
                "common_issues": ["field name mismatch", "different data types", "encoding issues"],
                "solutions": ["Check field names match exactly", "Verify field data types", "Check text encoding"]
            },
            "buffer": {
                "common_issues": ["invalid geometry", "projection issues", "negative buffer"],
                "solutions": ["Fix geometries first", "Ensure projected CRS", "Use positive buffer values"]
            },
            "clip": {
                "common_issues": ["geometry overlap", "CRS mismatch", "invalid geometries"],
                "solutions": ["Check layer overlap", "Reproject to same CRS", "Validate geometries"]
            },
            "raster": {
                "common_issues": ["projection mismatch", "nodata values", "large file size"],
                "solutions": ["Match raster projections", "Handle nodata properly", "Process in tiles"]
            }
        }

    def analyze_context(self, error_msg: str, code: str, operation_type: Optional[str] = None) -> Dict:
        """Analyze error context and provide contextual suggestions"""
        context = {
            "operation_type": operation_type,
            "code_complexity": self._assess_code_complexity(code),
            "data_operations": self._identify_data_operations(code),
            "contextual_solutions": []
        }

        # Add operation-specific solutions
        if operation_type and operation_type.lower() in self.operation_contexts:
            context["contextual_solutions"].extend(
                self.operation_contexts[operation_type.lower()]["solutions"]
            )

        # Analyze code patterns
        if "processing.run" in code and "Algorithm" in error_msg:
            context["contextual_solutions"].append("Check algorithm ID is correct and available")

        if "QgsVectorLayer" in code and "invalid" in error_msg.lower():
            context["contextual_solutions"].append("Verify layer path and check if layer.isValid()")

        return context

    def _assess_code_complexity(self, code: str) -> str:
        """Assess code complexity for debugging strategy"""
        lines = code.split('\n')
        if len(lines) < 10:
            return "simple"
        elif len(lines) < 30:
            return "moderate"
        else:
            return "complex"

    def _identify_data_operations(self, code: str) -> List[str]:
        """Identify types of data operations in code"""
        operations = []
        if "processing.run" in code:
            operations.append("processing_algorithm")
        if "QgsVectorLayer" in code:
            operations.append("vector_layer")
        if "QgsRasterLayer" in code:
            operations.append("raster_layer")
        if "join" in code.lower():
            operations.append("attribute_join")
        if "buffer" in code.lower():
            operations.append("buffer_operation")
        return operations


class AdaptiveLearning:
    """Handles learning from debugging history"""

    def __init__(self, history_file: str = None):
        self.history_file = history_file or "debug_history.json"
        self.history = self._load_history()

    def _load_history(self) -> Dict:
        """Load debugging history from file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                pass

        return {
            "successful_fixes": [],
            "failed_attempts": [],
            "common_patterns": {},
            "performance_metrics": {},
            "last_updated": None
        }

    def _save_history(self):
        """Save debugging history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Failed to save debug history: {e}")

    def record_debug_attempt(self, error_type: str, solution: str, success: bool,
                           execution_time: float = 0.0):
        """Record a debugging attempt for learning"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "solution": solution,
            "success": success,
            "execution_time": execution_time
        }

        if success:
            self.history["successful_fixes"].append(entry)
            # Update common patterns
            if error_type in self.history["common_patterns"]:
                self.history["common_patterns"][error_type] += 1
            else:
                self.history["common_patterns"][error_type] = 1
        else:
            self.history["failed_attempts"].append(entry)

        self.history["last_updated"] = datetime.now().isoformat()
        self._save_history()

    def get_best_solution(self, error_type: str) -> Optional[str]:
        """Get the most successful solution for an error type"""
        successful_solutions = {}

        for fix in self.history["successful_fixes"]:
            if fix["error_type"] == error_type:
                solution = fix["solution"]
                if solution in successful_solutions:
                    successful_solutions[solution] += 1
                else:
                    successful_solutions[solution] = 1

        if successful_solutions:
            return max(successful_solutions, key=successful_solutions.get)
        return None


class FallbackStrategy:
    """Manages fallback strategies for debugging"""

    def __init__(self):
        self.strategies = [
            {
                "name": "alternative_qgis_tool",
                "description": "Try alternative QGIS processing algorithm",
                "priority": 1,
                "conditions": ["algorithm_error", "qgis_specific"]
            },
            {
                "name": "geopandas_equivalent",
                "description": "Use geopandas/pandas equivalent operation",
                "priority": 2,
                "conditions": ["vector_operations", "data_processing"]
            },
            {
                "name": "break_into_steps",
                "description": "Break complex operation into simpler steps",
                "priority": 3,
                "conditions": ["complex_operation", "memory_error"]
            },
            {
                "name": "native_python",
                "description": "Use native Python libraries (numpy, scipy, etc.)",
                "priority": 4,
                "conditions": ["calculation_error", "analysis_error"]
            },
            {
                "name": "manual_implementation",
                "description": "Implement custom solution from scratch",
                "priority": 5,
                "conditions": ["no_suitable_tool", "specific_requirements"]
            }
        ]

    def get_fallback_strategies(self, error_category: str, operation_type: str = None) -> List[Dict]:
        """Get appropriate fallback strategies for error type"""
        applicable_strategies = []

        for strategy in self.strategies:
            if error_category in strategy["conditions"] or operation_type in strategy["conditions"]:
                applicable_strategies.append(strategy)

        # If no specific matches, return top 3 strategies
        if not applicable_strategies:
            applicable_strategies = self.strategies[:3]

        return sorted(applicable_strategies, key=lambda x: x["priority"])


class SmartDebugger:
    """Main smart debugging coordinator"""

    def __init__(self, history_file: str = None):
        self.pattern_matcher = ErrorPatternMatcher()
        self.context_analyzer = ContextAnalyzer()
        self.adaptive_learning = AdaptiveLearning(history_file)
        self.fallback_strategy = FallbackStrategy()

    def analyze_error(self, error_msg: str, code: str, operation_type: str = None) -> Dict:
        """Comprehensive error analysis"""
        # Pattern matching
        error_category, pattern_info = self.pattern_matcher.match_error_pattern(error_msg)

        # Context analysis
        context = self.context_analyzer.analyze_context(error_msg, code, operation_type)

        # Get historical best solution
        best_solution = None
        if error_category:
            best_solution = self.adaptive_learning.get_best_solution(error_category)

        # Get fallback strategies
        fallback_strategies = self.fallback_strategy.get_fallback_strategies(
            error_category or "unknown", operation_type
        )

        return {
            "error_category": error_category,
            "pattern_info": pattern_info,
            "context": context,
            "best_historical_solution": best_solution,
            "fallback_strategies": fallback_strategies,
            "confidence": self._calculate_confidence(error_category, pattern_info, best_solution)
        }

    def generate_debug_suggestions(self, error_msg: str, code: str, operation_type: str = None) -> List[str]:
        """Generate comprehensive debugging suggestions"""
        analysis = self.analyze_error(error_msg, code, operation_type)
        suggestions = []

        # Add pattern-based solutions
        if analysis["pattern_info"].get("solutions"):
            suggestions.extend(analysis["pattern_info"]["solutions"])

        # Add contextual solutions
        if analysis["context"]["contextual_solutions"]:
            suggestions.extend(analysis["context"]["contextual_solutions"])

        # Add historical best solution
        if analysis["best_historical_solution"]:
            suggestions.insert(0, f"Previously successful: {analysis['best_historical_solution']}")

        # Add fallback strategies if confidence is low
        if analysis["confidence"] < 0.7:
            for strategy in analysis["fallback_strategies"][:2]:
                suggestions.append(f"Fallback: {strategy['description']}")

        return suggestions

    def _calculate_confidence(self, error_category: str, pattern_info: Dict, best_solution: str) -> float:
        """Calculate confidence in debugging suggestions"""
        confidence = 0.0

        if error_category:
            confidence += 0.4
        if pattern_info.get("severity") == "high":
            confidence += 0.3
        if best_solution:
            confidence += 0.3

        return min(confidence, 1.0)

    def record_fix_attempt(self, error_msg: str, solution: str, success: bool, execution_time: float = 0.0):
        """Record the result of a debugging attempt"""
        error_category, _ = self.pattern_matcher.match_error_pattern(error_msg)
        if error_category:
            self.adaptive_learning.record_debug_attempt(
                error_category, solution, success, execution_time
            )


# Convenience function for quick debugging
def get_debug_suggestions(error_msg: str, code: str, operation_type: str = None) -> List[str]:
    """Quick function to get debugging suggestions"""
    debugger = SmartDebugger()
    return debugger.generate_debug_suggestions(error_msg, code, operation_type)


# Example usage and testing
if __name__ == "__main__":
    # Test the smart debugger
    debugger = SmartDebugger()

    test_error = "ModuleNotFoundError: No module named 'geopandas'"
    test_code = "import geopandas as gpd\ngdf = gpd.read_file('test.shp')"

    suggestions = debugger.generate_debug_suggestions(test_error, test_code, "data_loading")
    print("Debug suggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")