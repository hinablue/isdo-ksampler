"""
ISDO Test Suite
==============

This module contains comprehensive tests for the ISDO (Infinite Spectral Diffusion Odyssey) framework.

Test Files:
- test_isdo_integration.py: Basic ISDO framework integration tests
- test_isdo_core_integration.py: Core module integration tests
- test_ode_system_integration.py: VariationalODESystem integration tests

Usage:
    Run individual test files:
    python modules_forge/isdo/test/test_isdo_integration.py

    Or run all tests from project root:
    python -m pytest modules_forge/isdo/test/
"""

__version__ = "1.0.0"
__author__ = "ISDO Development Team"

# Import test modules for convenience
try:
    from .test_isdo_integration import *
    from .test_isdo_core_integration import *
    from .test_ode_system_integration import *
except ImportError:
    # Handle case where test dependencies are not available
    pass