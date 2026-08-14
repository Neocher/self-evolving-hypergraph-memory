"""/api/health 与 /api/version 端点测试 (红基线: 端点尚不存在)"""
from fastapi.testclient import TestClient
from api.app import create_app
from shm._version import __version__


def _client():
    return TestClient(create_app())


def test_health_endpoint():
    r = _client().get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "graph_connected" in data
    assert "faiss_loaded" in data


def test_version_endpoint():
    r = _client().get("/api/version")
    assert r.status_code == 200
    data = r.json()
    assert data["version"] == __version__
    assert "version_name" in data


def test_health_no_auth():
    # 只读探活端点不应要求身份
    r = _client().get("/api/health")
    assert r.status_code == 200
