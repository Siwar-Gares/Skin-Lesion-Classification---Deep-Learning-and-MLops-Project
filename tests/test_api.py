"""
Unit tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app

client = TestClient(app)


class TestAPIEndpoints:
    """Test FastAPI endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        assert 'classes' in data
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert data['status'] == 'healthy'
    
    def test_classes_endpoint(self):
        """Test classes endpoint"""
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert 'classes' in data
        assert len(data['classes']) == 7
    
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = client.get("/model_info")
        # May fail if model not loaded, but endpoint should exist
        assert response.status_code in [200, 503]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
