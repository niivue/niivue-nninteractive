"""
Pytest configuration and fixtures for vx-nninteractive API tests
"""
import os
import pytest
from fastapi.testclient import TestClient
from pathlib import Path


@pytest.fixture
def test_image_path():
    """Path to the test image"""
    # Use the same image that the frontend loads by default
    image_path = Path(__file__).parent.parent / "frontend" / "public" / "FLAIR.nii.gz"
    
    if not image_path.exists():
        pytest.skip(f"Test image not found at {image_path}")
    
    return str(image_path)


@pytest.fixture
def sample_scribbles():
    return [
        {"x": 110, "y": 132, "z": 9, "is_positive": True},
        {"x": 111, "y": 132, "z": 9, "is_positive": True},
        {"x": 111, "y": 133, "z": 9, "is_positive": True},
        {"x": 111, "y": 134, "z": 9, "is_positive": True},
        {"x": 111, "y": 135, "z": 9, "is_positive": True},
    ]


@pytest.fixture
def real_scribbles():
    return [
        {"x": 110, "y": 132, "z": 9, "is_positive": True},
        {"x": 111, "y": 132, "z": 9, "is_positive": True},
        {"x": 111, "y": 133, "z": 9, "is_positive": True},
        {"x": 111, "y": 134, "z": 9, "is_positive": True},
        {"x": 111, "y": 135, "z": 9, "is_positive": True},
        {"x": 111, "y": 136, "z": 9, "is_positive": True},
        {"x": 89, "y": 137, "z": 9, "is_positive": True},
        {"x": 111, "y": 137, "z": 9, "is_positive": True},
        {"x": 112, "y": 137, "z": 9, "is_positive": True},
        {"x": 89, "y": 138, "z": 9, "is_positive": True},
        {"x": 112, "y": 138, "z": 9, "is_positive": True},
        {"x": 89, "y": 139, "z": 9, "is_positive": True},
        {"x": 90, "y": 139, "z": 9, "is_positive": True},
        {"x": 112, "y": 139, "z": 9, "is_positive": True},
        {"x": 90, "y": 140, "z": 9, "is_positive": True},
        {"x": 112, "y": 140, "z": 9, "is_positive": True},
        {"x": 113, "y": 140, "z": 9, "is_positive": True},
        {"x": 90, "y": 141, "z": 9, "is_positive": True},
        {"x": 113, "y": 141, "z": 9, "is_positive": True},
        {"x": 90, "y": 142, "z": 9, "is_positive": True},
        {"x": 91, "y": 142, "z": 9, "is_positive": True},
        {"x": 113, "y": 142, "z": 9, "is_positive": True},
        {"x": 91, "y": 143, "z": 9, "is_positive": True},
        {"x": 114, "y": 143, "z": 9, "is_positive": True},
        {"x": 91, "y": 144, "z": 9, "is_positive": True},
        {"x": 114, "y": 144, "z": 9, "is_positive": True},
        {"x": 72, "y": 145, "z": 9, "is_positive": True},
        {"x": 91, "y": 145, "z": 9, "is_positive": True},
        {"x": 114, "y": 145, "z": 9, "is_positive": True},
        {"x": 73, "y": 146, "z": 9, "is_positive": True},
        {"x": 91, "y": 146, "z": 9, "is_positive": True},
        {"x": 114, "y": 146, "z": 9, "is_positive": True},
        {"x": 73, "y": 147, "z": 9, "is_positive": True},
        {"x": 92, "y": 147, "z": 9, "is_positive": True},
        {"x": 114, "y": 147, "z": 9, "is_positive": True},
        {"x": 73, "y": 148, "z": 9, "is_positive": True},
        {"x": 74, "y": 148, "z": 9, "is_positive": True},
        {"x": 92, "y": 148, "z": 9, "is_positive": True},
        {"x": 114, "y": 148, "z": 9, "is_positive": True},
        {"x": 74, "y": 149, "z": 9, "is_positive": True},
        {"x": 92, "y": 149, "z": 9, "is_positive": True},
        {"x": 115, "y": 149, "z": 9, "is_positive": True},
        {"x": 74, "y": 150, "z": 9, "is_positive": True},
        {"x": 92, "y": 150, "z": 9, "is_positive": True},
        {"x": 115, "y": 150, "z": 9, "is_positive": True},
        {"x": 74, "y": 151, "z": 9, "is_positive": True},
        {"x": 93, "y": 151, "z": 9, "is_positive": True},
        {"x": 115, "y": 151, "z": 9, "is_positive": True},
        {"x": 74, "y": 152, "z": 9, "is_positive": True},
        {"x": 93, "y": 152, "z": 9, "is_positive": True},
        {"x": 115, "y": 152, "z": 9, "is_positive": True},
        {"x": 75, "y": 153, "z": 9, "is_positive": True},
        {"x": 93, "y": 153, "z": 9, "is_positive": True},
        {"x": 116, "y": 153, "z": 9, "is_positive": True},
        {"x": 75, "y": 154, "z": 9, "is_positive": True},
        {"x": 94, "y": 154, "z": 9, "is_positive": True},
        {"x": 116, "y": 154, "z": 9, "is_positive": True},
        {"x": 75, "y": 155, "z": 9, "is_positive": True},
        {"x": 94, "y": 155, "z": 9, "is_positive": True},
        {"x": 116, "y": 155, "z": 9, "is_positive": True},
        {"x": 76, "y": 156, "z": 9, "is_positive": True},
        {"x": 94, "y": 156, "z": 9, "is_positive": True},
        {"x": 117, "y": 156, "z": 9, "is_positive": True},
        {"x": 76, "y": 157, "z": 9, "is_positive": True},
        {"x": 94, "y": 157, "z": 9, "is_positive": True},
        {"x": 117, "y": 157, "z": 9, "is_positive": True},
        {"x": 76, "y": 158, "z": 9, "is_positive": True},
        {"x": 94, "y": 158, "z": 9, "is_positive": True},
        {"x": 117, "y": 158, "z": 9, "is_positive": True},
        {"x": 77, "y": 159, "z": 9, "is_positive": True},
        {"x": 94, "y": 159, "z": 9, "is_positive": True},
        {"x": 95, "y": 159, "z": 9, "is_positive": True},
        {"x": 117, "y": 159, "z": 9, "is_positive": True},
        {"x": 77, "y": 160, "z": 9, "is_positive": True},
        {"x": 95, "y": 160, "z": 9, "is_positive": True},
        {"x": 118, "y": 160, "z": 9, "is_positive": True},
        {"x": 77, "y": 161, "z": 9, "is_positive": True},
        {"x": 95, "y": 161, "z": 9, "is_positive": True},
        {"x": 118, "y": 161, "z": 9, "is_positive": True},
        {"x": 77, "y": 162, "z": 9, "is_positive": True},
        {"x": 78, "y": 162, "z": 9, "is_positive": True},
        {"x": 95, "y": 162, "z": 9, "is_positive": True},
        {"x": 96, "y": 162, "z": 9, "is_positive": True},
        {"x": 118, "y": 162, "z": 9, "is_positive": True},
        {"x": 78, "y": 163, "z": 9, "is_positive": True},
        {"x": 96, "y": 163, "z": 9, "is_positive": True},
        {"x": 118, "y": 163, "z": 9, "is_positive": True},
        {"x": 78, "y": 164, "z": 9, "is_positive": True},
        {"x": 96, "y": 164, "z": 9, "is_positive": True},
        {"x": 119, "y": 164, "z": 9, "is_positive": True},
        {"x": 78, "y": 165, "z": 9, "is_positive": True},
        {"x": 96, "y": 165, "z": 9, "is_positive": True},
        {"x": 119, "y": 165, "z": 9, "is_positive": True},
        {"x": 79, "y": 166, "z": 9, "is_positive": True},
        {"x": 97, "y": 166, "z": 9, "is_positive": True},
        {"x": 119, "y": 166, "z": 9, "is_positive": True},
        {"x": 79, "y": 167, "z": 9, "is_positive": True},
        {"x": 97, "y": 167, "z": 9, "is_positive": True},
        {"x": 120, "y": 167, "z": 9, "is_positive": True},
        {"x": 79, "y": 168, "z": 9, "is_positive": True},
        {"x": 97, "y": 168, "z": 9, "is_positive": True},
        {"x": 120, "y": 168, "z": 9, "is_positive": True},
        {"x": 80, "y": 169, "z": 9, "is_positive": True},
        {"x": 97, "y": 169, "z": 9, "is_positive": True},
        {"x": 120, "y": 169, "z": 9, "is_positive": True},
        {"x": 80, "y": 170, "z": 9, "is_positive": True},
        {"x": 97, "y": 170, "z": 9, "is_positive": True},
        {"x": 120, "y": 170, "z": 9, "is_positive": True},
        {"x": 80, "y": 171, "z": 9, "is_positive": True},
        {"x": 121, "y": 171, "z": 9, "is_positive": True},
        {"x": 80, "y": 172, "z": 9, "is_positive": True},
        {"x": 121, "y": 172, "z": 9, "is_positive": True},
        {"x": 121, "y": 173, "z": 9, "is_positive": True},
    ]


@pytest.fixture
def client():
    """FastAPI test client"""
    # Import here to avoid circular imports and ensure proper initialization
    from src.niivue_nninteractive.api import app, initialize_model
    
    # Initialize the model for testing
    try:
        initialize_model()
    except Exception as e:
        pytest.skip(f"Could not initialize model for testing: {e}")
    
    return TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before running tests"""
    # Ensure we're in the right directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    yield
    
    # Cleanup after tests if needed...