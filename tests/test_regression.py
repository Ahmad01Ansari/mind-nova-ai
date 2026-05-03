from fastapi.testclient import TestClient
import sys, os
os.environ["FASTAPI_BRIDGE_SECRET"] = "mock_secret"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app
client = TestClient(app)
print("✅ Regression test setup complete. To run full suite, use: pytest -v")
