from fastapi.testclient import TestClient
import os, sys
os.environ["FASTAPI_BRIDGE_SECRET"] = "mock_secret"
from main import app
client = TestClient(app)

payload = {
    "userId": "test",
    "work_hours": 8.0,
    "sleep_hours": 8.0,
    "screen_time": 4.0,
    "break_frequency": 4.0,
    "stress_level": 3.0,
    "job_satisfaction": 8.0,
    "experience_years": 5.0
}
res = client.post("/predict/burnout", json=payload, headers={"x-bridge-secret": "mock_secret"})
print(res.json())
