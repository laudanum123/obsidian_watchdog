import pytest
import pathlib
from unittest.mock import MagicMock

@pytest.fixture
def patch_windows_path_methods(mocker):
    # Patch read_text and is_file for all WindowsPath instances
    mock_read_text = mocker.patch('pathlib.WindowsPath.read_text')
    mock_is_file = mocker.patch('pathlib.WindowsPath.is_file')
    return mock_read_text, mock_is_file 

@pytest.fixture
def comprehensive_path_mocking(mocker):
    """More comprehensive fixture for mocking Path-related operations.
    This handles multiple ways paths might be used."""
    
    # Create mocks for all the path methods we need to intercept
    mock_read_text = mocker.patch('pathlib.Path.read_text')
    mock_is_file = mocker.patch('pathlib.Path.is_file')
    mock_exists = mocker.patch('pathlib.Path.exists')
    
    # Also mock the WindowsPath variants
    mocker.patch('pathlib.WindowsPath.read_text', mock_read_text)
    mocker.patch('pathlib.WindowsPath.is_file', mock_is_file)
    mocker.patch('pathlib.WindowsPath.exists', mock_exists)
    
    # Also patch the Path constructor to ensure we're intercepting all instances
    original_path_new = pathlib.Path.__new__
    
    def patched_path_new(cls, *args, **kwargs):
        path_instance = original_path_new(cls, *args, **kwargs)
        # Ensure the instance has our mocked methods
        return path_instance
    
    mocker.patch('pathlib.Path.__new__', patched_path_new)
    
    # Return all the mocks for test usage
    return {
        'read_text': mock_read_text,
        'is_file': mock_is_file,
        'exists': mock_exists
    } 