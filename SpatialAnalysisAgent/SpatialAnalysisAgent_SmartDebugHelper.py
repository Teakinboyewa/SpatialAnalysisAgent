"""
Smart Debug Helper for SpatialAnalysisAgent
Integration helper to seamlessly use the smart debugging system within the existing workflow.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

# Import the smart debugger
try:
    from SpatialAnalysisAgent_SmartDebugger import SmartDebugger
except ImportError:
    # Fallback import path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from SpatialAnalysisAgent_SmartDebugger import SmartDebugger


class SmartDebugHelper:
    """Helper class to integrate smart debugging into the existing SpatialAnalysisAgent workflow"""

    def __init__(self):
        """Initialize the smart debug helper"""
        try:
            self.debugger = SmartDebugger()
            self.is_available = True
        except Exception as e:
            print(f"Warning: Smart debugger not available: {e}")
            self.debugger = None
            self.is_available = False

    def analyze_and_suggest(self, error_msg: str, code: str, operation_type: str = None) -> Dict:
        """
        Analyze error and provide suggestions using smart debugging

        Args:
            error_msg: The error message from the failed execution
            code: The code that failed
            operation_type: Optional operation type for context

        Returns:
            Dictionary with analysis results and suggestions
        """
        if not self.is_available:
            return self._fallback_analysis(error_msg, code)

        try:
            analysis = self.debugger.analyze_error(error_msg, code, operation_type)
            suggestions = self.debugger.generate_debug_suggestions(error_msg, code, operation_type)

            return {
                "success": True,
                "analysis": analysis,
                "suggestions": suggestions,
                "confidence": analysis.get("confidence", 0.0),
                "error_category": analysis.get("error_category", "unknown"),
                "fallback_strategies": analysis.get("fallback_strategies", [])
            }
        except Exception as e:
            print(f"Error in smart debugging: {e}")
            return self._fallback_analysis(error_msg, code)

    def _fallback_analysis(self, error_msg: str, code: str) -> Dict:
        """Fallback analysis when smart debugger is not available"""
        suggestions = []

        # Basic pattern matching
        if "ImportError" in error_msg or "ModuleNotFoundError" in error_msg:
            suggestions.extend([
                "Check if the required library is installed",
                "Try alternative import statements",
                "Ensure QGIS environment is properly set up"
            ])
        elif "QgsVectorLayer" in error_msg:
            suggestions.extend([
                "Check if the data path exists and is accessible",
                "Verify the layer format is supported",
                "Import QgsVectorLayer from qgis.core"
            ])
        elif "processing.run" in error_msg:
            suggestions.extend([
                "Check algorithm ID spelling",
                "Ensure all required parameters are provided",
                "Verify QGIS processing providers are enabled"
            ])
        else:
            suggestions.extend([
                "Check data paths and file permissions",
                "Verify code syntax and imports",
                "Consider using alternative approaches"
            ])

        return {
            "success": False,
            "analysis": {"error_category": "unknown", "confidence": 0.3},
            "suggestions": suggestions,
            "confidence": 0.3,
            "error_category": "unknown",
            "fallback_strategies": []
        }

    def record_fix_result(self, error_msg: str, solution: str, success: bool, execution_time: float = 0.0):
        """Record the result of a debugging attempt for learning"""
        if self.is_available:
            try:
                self.debugger.record_fix_attempt(error_msg, solution, success, execution_time)
            except Exception as e:
                print(f"Failed to record debug result: {e}")

    def get_debugging_prompt_enhancement(self, error_msg: str, code: str, operation_type: str = None) -> str:
        """Get enhanced debugging prompt with smart suggestions"""
        if not self.is_available:
            return ""

        try:
            suggestions = self.debugger.generate_debug_suggestions(error_msg, code, operation_type)
            if suggestions:
                enhancement = "\n\nSmart Debugging Suggestions:\n"
                for i, suggestion in enumerate(suggestions[:5], 1):  # Limit to top 5
                    enhancement += f"{i}. {suggestion}\n"
                return enhancement
        except Exception as e:
            print(f"Error generating prompt enhancement: {e}")

        return ""

    def get_enhanced_requirements(self, error_msg: str, code: str, operation_type: str = None) -> List[str]:
        """Get enhanced debugging requirements with smart suggestions"""
        base_requirements = [
            "Analyze the error pattern and apply contextual debugging strategies",
            "Elaborate your reasons for revision based on error analysis",
            "Return the entire corrected program in one Python code block",
            "Use step-by-step approach for multi-operation tasks",
            "Consider alternative tools and libraries if QGIS tools fail"
        ]

        if not self.is_available:
            return base_requirements

        try:
            suggestions = self.debugger.generate_debug_suggestions(error_msg, code, operation_type)
            enhanced_requirements = base_requirements + [f"Smart Debug: {s}" for s in suggestions[:3]]
            return enhanced_requirements
        except Exception as e:
            print(f"Error getting enhanced requirements: {e}")
            return base_requirements

    def format_error_context(self, error_msg: str, code: str, operation_type: str = None) -> str:
        """Format error context for better debugging prompt"""
        context = f"Error Message: {error_msg}\n\n"
        context += f"Failed Code:\n{code}\n\n"

        if operation_type:
            context += f"Operation Type: {operation_type}\n\n"

        if self.is_available:
            try:
                analysis = self.debugger.analyze_error(error_msg, code, operation_type)
                if analysis.get("error_category"):
                    context += f"Error Category: {analysis['error_category']}\n"
                if analysis.get("confidence"):
                    context += f"Analysis Confidence: {analysis['confidence']:.2f}\n"
            except Exception as e:
                print(f"Error formatting context: {e}")

        return context

    def get_fallback_strategy(self, error_category: str, operation_type: str = None) -> Optional[str]:
        """Get appropriate fallback strategy for error type"""
        if not self.is_available:
            return "Try alternative tools or simpler implementation approaches"

        try:
            analysis = self.debugger.analyze_error("", "", operation_type)
            strategies = analysis.get("fallback_strategies", [])
            if strategies:
                return strategies[0].get("description", "")
        except Exception as e:
            print(f"Error getting fallback strategy: {e}")

        return "Try alternative tools or simpler implementation approaches"


# Global instance for easy access
_smart_debug_helper = None

def get_smart_debug_helper():
    """Get global smart debug helper instance"""
    global _smart_debug_helper
    if _smart_debug_helper is None:
        _smart_debug_helper = SmartDebugHelper()
    return _smart_debug_helper


# Convenience functions for direct use
def analyze_error_smart(error_msg: str, code: str, operation_type: str = None) -> Dict:
    """Quick function to analyze error with smart debugging"""
    helper = get_smart_debug_helper()
    return helper.analyze_and_suggest(error_msg, code, operation_type)


def get_smart_suggestions(error_msg: str, code: str, operation_type: str = None) -> List[str]:
    """Quick function to get smart debugging suggestions"""
    helper = get_smart_debug_helper()
    result = helper.analyze_and_suggest(error_msg, code, operation_type)
    return result.get("suggestions", [])


def enhance_debug_prompt(error_msg: str, code: str, operation_type: str = None) -> str:
    """Quick function to enhance debugging prompt"""
    helper = get_smart_debug_helper()
    return helper.get_debugging_prompt_enhancement(error_msg, code, operation_type)


def record_debug_outcome(error_msg: str, solution: str, success: bool, execution_time: float = 0.0):
    """Quick function to record debugging outcome for learning"""
    helper = get_smart_debug_helper()
    helper.record_fix_result(error_msg, solution, success, execution_time)


# Example usage and testing
if __name__ == "__main__":
    # Test the smart debug helper
    helper = SmartDebugHelper()

    test_error = "ImportError: No module named 'geopandas'"
    test_code = "import geopandas as gpd\ngdf = gpd.read_file('test.shp')"

    result = helper.analyze_and_suggest(test_error, test_code, "data_loading")
    print("Smart Debug Analysis:")
    print(f"Error Category: {result['error_category']}")
    print(f"Confidence: {result['confidence']}")
    print("Suggestions:")
    for i, suggestion in enumerate(result['suggestions'], 1):
        print(f"  {i}. {suggestion}")

    # Test prompt enhancement
    enhancement = helper.get_debugging_prompt_enhancement(test_error, test_code)
    print(f"\nPrompt Enhancement: {enhancement}")