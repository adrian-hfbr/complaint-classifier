# tests/test_api.py
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from src.main import app
from src.services import ModelService


def test_classify_complaint_success(mocker):
    """
    Tests the /classify endpoint for a successful prediction.
    """
    # Arrange: Create a mock and set its return value
    mock_service = MagicMock(spec=ModelService)
    mock_service.predict.return_value = "Credit card or prepaid card"

    # Use mocker.patch to replace the global 'model_service' in src.main
    # with the mock.
    mocker.patch('src.main.model_service', mock_service)
    
    client = TestClient(app)
    
    # Act
    test_narrative = "My credit card was charged twice for the same item."
    response = client.post("/classify", json={"narrative": test_narrative})

    # Assert
    assert response.status_code == 200
    assert response.json()["predicted_product"] == "Credit card or prepaid card"
    mock_service.predict.assert_called_with(test_narrative)


def test_classify_complaint_invalid_input():
    """
    Tests that the API returns a 422 error for invalid input.
    No mocking is needed because this is handled by FastAPI before the service is called.
    """
    client = TestClient(app)
    response = client.post("/classify", json={"narrative": "short"})
    assert response.status_code == 422


def test_classify_complaint_server_error(mocker):
    """
    Tests that the API returns a 500 error if the model service fails.
    """
    # Arrange: Create a mock and configure it to raise an exception
    mock_service = MagicMock(spec=ModelService)
    mock_service.predict.side_effect = Exception("Model failed")
    
    # Patch the global 'model_service' in src.main with the error-raising mock
    mocker.patch('src.main.model_service', mock_service)
    
    client = TestClient(app)

    # Act
    response = client.post("/classify", json={"narrative": "This will cause an error."})
    
    # Assert
    assert response.status_code == 500
    assert response.json() == {"detail": "Model inference failed."}