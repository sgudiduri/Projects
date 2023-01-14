import math
import numpy as np
from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    # Given
    
    # When
    response = client.get(
        "http://localhost:8001/api/v1/health"
    )

    # Then
    assert response.status_code == 200
    prediction_data = response.json()
    assert prediction_data["errors"] is None
    

###############################################
### TO DO 
### ADD MORE TESTS
###############################################