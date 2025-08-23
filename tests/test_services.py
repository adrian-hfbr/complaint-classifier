# tests/test_services.py
import pytest
from pathlib import Path
from unittest.mock import MagicMock
import joblib

from src.services import ModelService

@pytest.fixture
def dummy_pipeline_path(tmp_path: Path) -> Path:
    """Creates a dummy pipeline file and returns its path."""
    file_path = tmp_path / "test_pipeline.pkl"
    file_path.touch()
    return file_path

def test_model_service_init_success(dummy_pipeline_path: Path, mocker):
    """
    Tests that the ModelService initializes correctly when the pipeline file exists.
    """
    # Arrange
    mock_pipeline = MagicMock()
    # Use mocker.patch to replace joblib.load ONLY for this test
    mocker.patch('joblib.load', return_value=mock_pipeline)

    # Act
    service = ModelService(pipeline_path=dummy_pipeline_path)

    # Assert
    assert service.pipeline is not None
    joblib.load.assert_called_once_with(dummy_pipeline_path)

def test_model_service_init_file_not_found():
    """
    Tests that the ModelService raises a FileNotFoundError if the artifact is missing.
    This test now runs with the REAL joblib.load, so it will raise the error correctly.
    """
    # Arrange
    non_existent_path = Path("non_existent_pipeline.pkl")

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        ModelService(pipeline_path=non_existent_path)

def test_model_service_predict(dummy_pipeline_path: Path, mocker):
    """
    Tests the predict method to ensure it formats the input and output correctly.
    """
    # Arrange
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = ['Credit card or prepaid card']
    mocker.patch('joblib.load', return_value=mock_pipeline)
    
    service = ModelService(pipeline_path=dummy_pipeline_path)
    test_narrative = "This is a test complaint about my credit card."

    # Act
    prediction = service.predict(test_narrative)

    # Assert
    mock_pipeline.predict.assert_called_once_with([test_narrative])
    assert prediction == 'Credit card or prepaid card'