"""
Unit and regression test for the EnergyExtraction package.
"""

# Import package, test suite, and other packages as needed
import sys
import pytest
import EnergyExtraction
from EnergyExtraction.example_api import example_function


def test_EnergyExtraction_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "EnergyExtraction" in sys.modules


def test_mdanalysis_logo_length(mdanalysis_logo_text):
    """Example test using a fixture defined in conftest.py"""
    logo_lines = mdanalysis_logo_text.split("\n")
    assert len(logo_lines) == 46, "Logo file does not have 46 lines!"

@pytest.mark.parametrize(
    "value, expected",
    [
        (0, 0),
        (1, 2),
        (-5, -10),
        (1.5, 3.0),
        (-2.0, -4.0)
    ],
)
def test_double_value(value, expected):
    assert example_function(value) == expected
