def test_recommend_valid(client):
    response = client.post(
        "/recommend", json={"input_param": numeric_features[0], "input_value": 10.0}
    )
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert data["recommendations"][numeric_features[0]] == 10.0

def test_recommend_invalid_param(client):
    response = client.post(
        "/recommend", json={"input_param": "invalid", "input_value": 5}
    )
    assert response.status_code == 400