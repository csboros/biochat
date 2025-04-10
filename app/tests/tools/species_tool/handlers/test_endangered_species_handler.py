"""Unit tests for the EndangeredSpeciesHandler class."""

import os
from unittest.mock import patch
from pathlib import Path
import toml
import pytest
from app.tools.species_tool.handlers.endangered_species_handler import EndangeredSpeciesHandler

@pytest.fixture(autouse=True)
def mock_env():
    """Mock environment variables for testing using values from secrets.toml."""
    secrets_path = Path(".streamlit/secrets.toml")
    secrets = toml.load(secrets_path)
    project_id = secrets.get("GOOGLE_CLOUD_PROJECT")
    with patch.dict(os.environ, {'GOOGLE_CLOUD_PROJECT': project_id}):
        yield

def test_get_occurrences_edge_cases():
    """Test edge cases and error handling for the get_occurrences method."""
    handler = EndangeredSpeciesHandler()

    # Test None input
    with pytest.raises(ValueError, match="Content cannot be None"):
        handler.get_occurrences(None)

    # Test invalid type
    with pytest.raises(TypeError, match="Expected dict or str, got <class 'int'>"):
        handler.get_occurrences(123)

    # Test empty dictionary
    with pytest.raises(ValueError, match="Content dictionary cannot be empty"):
        handler.get_occurrences({})

    # Test empty string
    with pytest.raises(ValueError, match="Content string cannot be empty or whitespace"):
        handler.get_occurrences("")

    # Test whitespace string
    with pytest.raises(ValueError, match="Content string cannot be empty or whitespace"):
        handler.get_occurrences("   ")

    # Test malformed dictionary
    with pytest.raises(ValueError):
        handler.get_occurrences({"invalid_key": "value"})

    # Test special characters
    special_chars = "🦁🐯🐻"
    with pytest.raises(ValueError, match="Invalid characters in content"):
        handler.get_occurrences(special_chars)

def test_get_occurrences_valid_cases():
    """Test valid cases for the get_occurrences method."""
    handler = EndangeredSpeciesHandler()

    # Test valid dictionary input
    valid_dict = {
        "species_name": "Panthera tigris",
        "location": "Asia"
    }
    result = handler.get_occurrences(valid_dict)
    assert isinstance(result, dict)
    assert "occurrences" in result

    # Test valid string input
    valid_string = "Panthera tigris"
    result = handler.get_occurrences(valid_string)
    assert isinstance(result, dict)
    assert "occurrences" in result