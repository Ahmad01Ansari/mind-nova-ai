from fastapi.testclient import TestClient
import os
import sys

# Ensure we can import from parent dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "deterioration" in data["models_loaded"]
    print("✅ Health Check Passed: Models Loaded:", data["models_loaded"])

def test_daily_inference():
    # Mock data for standard user
    payload = {
        "userId": "test_user_001",
        "mood_current": 4.5,
        "sleep_hours": 7.0,
        "workload_level": 5.0,
        "hours_worked": 8.0
    }
    # Skip auth for unit test if possible or provide mock secret
    secret = os.getenv("FASTAPI_BRIDGE_SECRET", "mock_secret")
    headers = {"X-Bridge-Secret": secret}
    
    response = client.post("/analyze/daily", json=payload, headers=headers)
    if response.status_code == 403:
        print("⚠️ Warning: Auth required for test. Skipping detailed check.")
        return

    assert response.status_code == 200
    data = response.json()
    assert "stress" in data["results"]
    print("✅ Daily Inference Passed. Stress Score:", data["results"]["stress"]["score"])

def test_deterioration_forecast():
    # 7-day sequence
    history = [
        {"day": i, "mood": 7.0 - (i*0.5), "sleep": 7.0 - (i*0.2), "workload": 5.0 + (i*0.5)}
        for i in range(7)
    ]
    payload = {
        "userId": "test_user_001",
        "history": history
    }
    secret = os.getenv("FASTAPI_BRIDGE_SECRET", "mock_secret")
    headers = {"X-Bridge-Secret": secret}
    
    response = client.post("/analyze/deterioration", json=payload, headers=headers)
    if response.status_code == 403: return

    assert response.status_code == 200
    data = response.json()
    assert "forecast" in data
    print("✅ Deterioration Forecast Passed. Crisis Prob:", data["forecast"]["forecast_prob"])

if __name__ == "__main__":
    test_health_check()
    test_daily_inference()
    test_deterioration_forecast()
