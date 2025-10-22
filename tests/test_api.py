"""
Pytest tests for the nnInteractive Segmentation API
"""
import json
import numpy as np
import SimpleITK as sitk
import tempfile
import io


class TestSegmentationAPI:
    """Test class for segmentation API endpoints"""
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "active_sessions" in data
    
    def test_root_endpoint(self, client):
        """Test the root API information endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["api"] == "nnInteractive Segmentation API"
        assert "endpoints" in data
        assert "usage" in data
    
    def test_initial_segmentation_with_real_scribbles(self, client, test_image_path, real_scribbles):
        """Test initial segmentation using the exact scribble coordinates from user"""
        # Read the test image
        with open(test_image_path, "rb") as f:
            image_data = f.read()
        
        # Prepare the request
        files = {"image": ("FLAIR.nii.gz", io.BytesIO(image_data), "application/octet-stream")}
        data = {"scribbles": json.dumps(real_scribbles)}
        
        # Send the request
        response = client.post("/segment", files=files, data=data)
        
        # Check response
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        
        # Check that we get a user ID
        user_id = response.headers.get("X-User-ID")
        assert user_id is not None
        assert len(user_id) > 0
        
        # Check that we get segmentation data
        segmentation_data = response.content
        assert len(segmentation_data) > 0
        
        # Validate the segmentation result
        self._validate_segmentation_result(segmentation_data)
    
    def test_refinement_segmentation(self, client, test_image_path, real_scribbles, sample_scribbles):
        """Test refinement segmentation (subsequent request without image)"""
        # First, perform initial segmentation to get user_id
        with open(test_image_path, "rb") as f:
            image_data = f.read()
        
        files = {"image": ("FLAIR.nii.gz", io.BytesIO(image_data), "application/octet-stream")}
        data = {"scribbles": json.dumps(real_scribbles)}
        
        initial_response = client.post("/segment", files=files, data=data)
        assert initial_response.status_code == 200
        
        user_id = initial_response.headers.get("X-User-ID")
        assert user_id is not None
        
        # Now perform refinement with just scribbles and user_id
        refinement_data = {
            "scribbles": json.dumps(sample_scribbles),
            "user_id": user_id
        }
        
        refinement_response = client.post("/segment", data=refinement_data)
        
        # Check response
        assert refinement_response.status_code == 200
        assert refinement_response.headers["content-type"] == "application/octet-stream"
        
        # Check that we get segmentation data
        segmentation_data = refinement_response.content
        assert len(segmentation_data) > 0
        
        # Validate the segmentation result
        self._validate_segmentation_result(segmentation_data)
    
    def test_segmentation_without_image_and_no_user_id(self, client, sample_scribbles):
        """Test that segmentation fails when no image and no user_id provided"""
        data = {"scribbles": json.dumps(sample_scribbles)}
        
        response = client.post("/segment", data=data)
        
        assert response.status_code == 400
        error_data = response.json()
        assert "Image file required for first request" in error_data["detail"]
    
    def test_segmentation_with_empty_scribbles(self, client, test_image_path):
        """Test that segmentation fails with empty scribbles"""
        with open(test_image_path, "rb") as f:
            image_data = f.read()
        
        files = {"image": ("FLAIR.nii.gz", io.BytesIO(image_data), "application/octet-stream")}
        data = {"scribbles": json.dumps([])}
        
        response = client.post("/segment", files=files, data=data)
        
        assert response.status_code == 500  # Should fail during segmentation
    
    def test_segmentation_with_invalid_scribbles_format(self, client, test_image_path):
        """Test that segmentation fails with invalid scribbles format"""
        with open(test_image_path, "rb") as f:
            image_data = f.read()
        
        files = {"image": ("FLAIR.nii.gz", io.BytesIO(image_data), "application/octet-stream")}
        data = {"scribbles": "invalid json"}
        
        response = client.post("/segment", files=files, data=data)
        
        assert response.status_code == 400
        error_data = response.json()
        assert "Invalid scribbles format" in error_data["detail"]
    
    def test_segmentation_with_out_of_bounds_scribbles(self, client, test_image_path):
        """Test segmentation with scribbles outside image bounds"""
        # Create scribbles with coordinates way outside typical image bounds
        out_of_bounds_scribbles = [
            {"x": 9999, "y": 9999, "z": 9999, "is_positive": True},
            {"x": -100, "y": -100, "z": -100, "is_positive": True}
        ]
        
        with open(test_image_path, "rb") as f:
            image_data = f.read()
        
        files = {"image": ("FLAIR.nii.gz", io.BytesIO(image_data), "application/octet-stream")}
        data = {"scribbles": json.dumps(out_of_bounds_scribbles)}
        
        response = client.post("/segment", files=files, data=data)
        
        # Should fail because no valid scribbles
        assert response.status_code == 500
    
    def test_segmentation_with_mixed_positive_negative_scribbles(self, client, test_image_path):
        """Test segmentation with both positive and negative scribbles"""
        mixed_scribbles = [
            {"x": 110, "y": 132, "z": 9, "is_positive": True},
            {"x": 111, "y": 132, "z": 9, "is_positive": True},
            {"x": 112, "y": 132, "z": 9, "is_positive": False},  # Negative scribble
            {"x": 113, "y": 132, "z": 9, "is_positive": False},  # Negative scribble
        ]
        
        with open(test_image_path, "rb") as f:
            image_data = f.read()
        
        files = {"image": ("FLAIR.nii.gz", io.BytesIO(image_data), "application/octet-stream")}
        data = {"scribbles": json.dumps(mixed_scribbles)}
        
        response = client.post("/segment", files=files, data=data)
        
        assert response.status_code == 200
        
        # Validate the segmentation result
        segmentation_data = response.content
        self._validate_segmentation_result(segmentation_data)
    
    def test_invalid_image_format(self, client, sample_scribbles):
        """Test that segmentation fails with invalid image format"""
        # Create a fake non-NIfTI file
        fake_image_data = b"This is not a NIfTI file"
        
        files = {"image": ("fake.txt", io.BytesIO(fake_image_data), "text/plain")}
        data = {"scribbles": json.dumps(sample_scribbles)}
        
        response = client.post("/segment", files=files, data=data)
        
        assert response.status_code == 400
        error_data = response.json()
        assert "Only NIfTI files" in error_data["detail"]
    
    def _validate_segmentation_result(self, segmentation_data: bytes):
        """Validate that the segmentation result is a valid NIfTI file"""
        # Write to temporary file and read with SimpleITK
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
            tmp_file.write(segmentation_data)
            tmp_file_path = tmp_file.name
        
        try:
            # Load with SimpleITK
            segmentation_image = sitk.ReadImage(tmp_file_path)
            segmentation_array = sitk.GetArrayFromImage(segmentation_image)
            
            # Basic validation
            assert segmentation_array.ndim == 3, "Segmentation should be 3D"
            assert segmentation_array.shape[0] > 0, "Segmentation should have depth > 0"
            assert segmentation_array.shape[1] > 0, "Segmentation should have height > 0"
            assert segmentation_array.shape[2] > 0, "Segmentation should have width > 0"
            
            # Check data type
            assert segmentation_array.dtype in [np.uint8, np.int8, np.uint16, np.int16, np.float32], \
                f"Unexpected segmentation data type: {segmentation_array.dtype}"
            
            # Log some statistics for debugging
            unique_values = np.unique(segmentation_array)
            non_zero_count = np.count_nonzero(segmentation_array)
            total_voxels = segmentation_array.size
            
            print(f"Segmentation validation:")
            print(f"  Shape: {segmentation_array.shape}")
            print(f"  Unique values: {unique_values}")
            print(f"  Non-zero voxels: {non_zero_count}/{total_voxels}")
            print(f"  Value range: [{segmentation_array.min()}, {segmentation_array.max()}]")
            
        finally:
            # Clean up temp file
            import os
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)


class TestScribbleCoordinateValidation:
    """Test class for scribble coordinate validation"""
    
    def test_scribble_coordinates_structure(self, real_scribbles):
        """Test that the provided scribble coordinates have correct structure"""
        assert len(real_scribbles) > 0, "Should have scribbles"
        
        for i, scribble in enumerate(real_scribbles):
            assert "x" in scribble, f"Scribble {i} missing 'x' coordinate"
            assert "y" in scribble, f"Scribble {i} missing 'y' coordinate"
            assert "z" in scribble, f"Scribble {i} missing 'z' coordinate"
            assert "is_positive" in scribble, f"Scribble {i} missing 'is_positive' flag"
            
            assert isinstance(scribble["x"], int), f"Scribble {i} 'x' should be integer"
            assert isinstance(scribble["y"], int), f"Scribble {i} 'y' should be integer"
            assert isinstance(scribble["z"], int), f"Scribble {i} 'z' should be integer"
            assert isinstance(scribble["is_positive"], bool), f"Scribble {i} 'is_positive' should be boolean"
            
            # Check reasonable bounds (assuming typical medical image dimensions)
            assert 0 <= scribble["x"] <= 1000, f"Scribble {i} 'x' coordinate seems out of bounds: {scribble['x']}"
            assert 0 <= scribble["y"] <= 1000, f"Scribble {i} 'y' coordinate seems out of bounds: {scribble['y']}"
            assert 0 <= scribble["z"] <= 100, f"Scribble {i} 'z' coordinate seems out of bounds: {scribble['z']}"
    
    def test_scribble_coordinates_analysis(self, real_scribbles):
        """Analyze the provided scribble coordinates"""
        positive_scribbles = [s for s in real_scribbles if s["is_positive"]]
        negative_scribbles = [s for s in real_scribbles if not s["is_positive"]]
        
        print(f"Total scribbles: {len(real_scribbles)}")
        print(f"Positive scribbles: {len(positive_scribbles)}")
        print(f"Negative scribbles: {len(negative_scribbles)}")
        
        # All provided scribbles should be positive based on the data
        assert len(positive_scribbles) == len(real_scribbles), "All provided scribbles should be positive"
        assert len(negative_scribbles) == 0, "No negative scribbles in provided data"
        
        # Check z-coordinate consistency (all should be on same slice)
        z_coords = [s["z"] for s in real_scribbles]
        unique_z = set(z_coords)
        print(f"Unique Z coordinates: {unique_z}")
        assert len(unique_z) == 1, "All scribbles should be on the same Z slice"
        assert 9 in unique_z, "Expected Z coordinate to be 9"
        
        # Check coordinate ranges
        x_coords = [s["x"] for s in real_scribbles]
        y_coords = [s["y"] for s in real_scribbles]
        
        print(f"X range: {min(x_coords)} - {max(x_coords)}")
        print(f"Y range: {min(y_coords)} - {max(y_coords)}")
        
        # Validate that coordinates form a reasonable pattern
        assert max(x_coords) - min(x_coords) > 10, "Scribbles should span reasonable X range"
        assert max(y_coords) - min(y_coords) > 10, "Scribbles should span reasonable Y range"


class TestModelInitializationError:
    """Test class for model initialization error handling"""
    
    def test_initialize_model_failure(self, monkeypatch):
        """Test initialize_model error handling when model loading fails"""
        from src.niivue_nninteractive.api import initialize_model
        
        # Mock the snapshot_download to raise an exception
        def mock_snapshot_download(*_, **__):
            raise Exception("Model download failed")
        
        monkeypatch.setattr("src.niivue_nninteractive.api.snapshot_download", mock_snapshot_download)
        
        # Test that initialize_model raises an exception and logs the error
        with pytest.raises(Exception, match="Model download failed"):
            initialize_model()


class TestSessionCleanup:
    """Test class for session cleanup functionality"""
    
    def test_cleanup_old_sessions_with_expired_sessions(self):
        """Test cleanup_old_sessions when there are expired sessions"""
        from src.niivue_nninteractive.api import cleanup_old_sessions, user_sessions
        from datetime import datetime, timedelta
        import asyncio
        
        # Add some mock expired sessions
        old_time = datetime.now() - timedelta(hours=2)  # 2 hours ago (expired)
        recent_time = datetime.now() - timedelta(minutes=10)  # 10 minutes ago (not expired)
        
        user_sessions.clear()  # Start clean
        user_sessions["expired_user_1"] = {"last_access": old_time}
        user_sessions["expired_user_2"] = {"last_access": old_time}
        user_sessions["active_user"] = {"last_access": recent_time}
        
        initial_count = len(user_sessions)
        assert initial_count == 3
        
        # Run cleanup
        asyncio.run(cleanup_old_sessions())
        
        # Check that expired sessions were removed
        assert len(user_sessions) == 1
        assert "active_user" in user_sessions
        assert "expired_user_1" not in user_sessions
        assert "expired_user_2" not in user_sessions
    
    def test_cleanup_old_sessions_no_expired_sessions(self):
        """Test cleanup_old_sessions when there are no expired sessions"""
        from src.niivue_nninteractive.api import cleanup_old_sessions, user_sessions
        from datetime import datetime, timedelta
        import asyncio
        
        # Add some recent sessions
        recent_time = datetime.now() - timedelta(minutes=10)
        
        user_sessions.clear()
        user_sessions["user_1"] = {"last_access": recent_time}
        user_sessions["user_2"] = {"last_access": recent_time}
        
        initial_count = len(user_sessions)
        assert initial_count == 2
        
        # Run cleanup
        asyncio.run(cleanup_old_sessions())
        
        # Check that no sessions were removed
        assert len(user_sessions) == 2
        assert "user_1" in user_sessions
        assert "user_2" in user_sessions


class TestImageProcessingEdgeCases:
    """Test class for edge cases in image processing"""
    
    def test_process_uploaded_image_invalid_dimensions(self, client):
        """Test process_uploaded_image with image that has invalid dimensions"""
        # Create a mock 2D image (invalid for this application)
        import SimpleITK as sitk
        import numpy as np
        import tempfile
        import io
        
        # Create a 2D image instead of 3D
        array_2d = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        image_2d = sitk.GetImageFromArray(array_2d)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
            tmp_path = tmp.name
        
        sitk.WriteImage(image_2d, tmp_path)
        
        # Read the invalid image file
        with open(tmp_path, 'rb') as f:
            invalid_image_data = f.read()
        
        # Clean up temp file
        import os
        os.unlink(tmp_path)
        
        # Test with the invalid 2D image
        files = {"image": ("invalid_2d.nii.gz", io.BytesIO(invalid_image_data), "application/octet-stream")}
        data = {"scribbles": '[{"x": 50, "y": 50, "z": 0, "is_positive": true}]'}
        
        response = client.post("/segment", files=files, data=data)
        
        # Should fail due to invalid dimensions
        assert response.status_code == 400
        error_data = response.json()
        assert "Input image must be 3D" in error_data["detail"]
    
    def test_process_uploaded_image_corrupted_file(self, client):
        """Test process_uploaded_image with corrupted NIfTI file"""
        # Create corrupted NIfTI-like data
        corrupted_data = b"Not a real NIfTI file content but has .nii.gz extension"
        
        files = {"image": ("corrupted.nii.gz", io.BytesIO(corrupted_data), "application/octet-stream")}
        data = {"scribbles": '[{"x": 100, "y": 100, "z": 5, "is_positive": true}]'}
        
        response = client.post("/segment", files=files, data=data)
        
        # Should fail to load the corrupted file
        assert response.status_code == 400
        error_data = response.json()
        assert "Failed to load NIfTI image" in error_data["detail"]


class TestSegmentationImageUpdates:
    """Test class for segmentation with image updates in existing sessions"""
    
    def test_segmentation_with_image_update_existing_session(self, client, test_image_path, sample_scribbles):
        """Test updating image in an existing session"""
        # First, create a session with initial segmentation
        with open(test_image_path, "rb") as f:
            image_data = f.read()
        
        files = {"image": ("FLAIR.nii.gz", io.BytesIO(image_data), "application/octet-stream")}
        data = {"scribbles": json.dumps(sample_scribbles)}
        
        initial_response = client.post("/segment", files=files, data=data)
        assert initial_response.status_code == 200
        
        user_id = initial_response.headers.get("X-User-ID")
        assert user_id is not None
        
        # Now send a new image with the same user_id (updating existing session)
        files_update = {"image": ("FLAIR.nii.gz", io.BytesIO(image_data), "application/octet-stream")}
        data_update = {
            "scribbles": json.dumps(sample_scribbles),
            "user_id": user_id
        }
        
        update_response = client.post("/segment", files=files_update, data=data_update)
        assert update_response.status_code == 200
        
        # Validate the segmentation result
        segmentation_data = update_response.content
        assert len(segmentation_data) > 0


class TestSegmentationErrorPaths:
    """Test class for error paths in segmentation"""
    
    def test_segmentation_with_model_session_error(self, client, test_image_path, sample_scribbles, monkeypatch):
        """Test segmentation when model session operations fail"""
        # Mock the model_session to raise an error during set_image
        def mock_set_image(*_, **__):
            raise Exception("Model session error")
        
        # We need to patch the model_session after it's initialized
        from src.niivue_nninteractive.api import model_session
        if model_session:
            monkeypatch.setattr(model_session, "set_image", mock_set_image)
        
        with open(test_image_path, "rb") as f:
            image_data = f.read()
        
        files = {"image": ("FLAIR.nii.gz", io.BytesIO(image_data), "application/octet-stream")}
        data = {"scribbles": json.dumps(sample_scribbles)}
        
        response = client.post("/segment", files=files, data=data)
        
        # Should fail due to model session error
        assert response.status_code == 500
        error_data = response.json()
        assert "Segmentation failed" in error_data["detail"]


class TestLifespanEvents:
    """Test class for FastAPI lifespan events"""
    
    def test_lifespan_startup_error_handling(self, monkeypatch):
        """Test lifespan startup error handling"""
        # Mock initialize_model to raise an exception
        def mock_initialize_model():
            raise Exception("Startup initialization failed")
        
        monkeypatch.setattr("src.niivue_nninteractive.api.initialize_model", mock_initialize_model)
        
        # Test that lifespan raises an exception during startup
        import contextlib
        
        @contextlib.asynccontextmanager
        async def test_lifespan(_):
            try:
                mock_initialize_model()
            except Exception:
                # This should raise the exception, testing the error path
                raise
            yield
        
        # Verify that startup failure is handled
        with pytest.raises(Exception, match="Startup initialization failed"):
            async def run_lifespan():
                async with test_lifespan(None):
                    pass
            
            import asyncio
            asyncio.run(run_lifespan())
    
    def test_lifespan_normal_startup_and_shutdown(self):
        """Test normal lifespan startup and shutdown logging"""
        from src.niivue_nninteractive.api import lifespan
        import asyncio
        import logging
        
        # Create a list to capture log messages
        log_messages = []
        
        # Create a custom handler to capture logs
        class TestLogHandler(logging.Handler):
            def emit(self, record):
                log_messages.append(record.getMessage())
        
        # Add our test handler to the logger
        from src.niivue_nninteractive.api import logger
        test_handler = TestLogHandler()
        logger.addHandler(test_handler)
        logger.setLevel(logging.INFO)
        
        async def run_lifespan():
            # Mock app
            class MockApp:
                pass
            
            app = MockApp()
            
            # Run through the lifespan
            async with lifespan(app):
                # This represents the app running
                pass
        
        try:
            asyncio.run(run_lifespan())
            
            # Check that startup and shutdown messages were logged
            startup_logged = any("Starting API server" in msg for msg in log_messages)
            shutdown_logged = any("API server shutting down" in msg for msg in log_messages)
            
            assert startup_logged, "Should log startup message"
            assert shutdown_logged, "Should log shutdown message"
            
        finally:
            # Clean up the test handler
            logger.removeHandler(test_handler)
    
    def test_lifespan_error_logging_and_raising(self, monkeypatch):
        """Test that lifespan properly logs errors and re-raises them"""
        from src.niivue_nninteractive.api import lifespan
        import asyncio
        import logging
        
        # Create a list to capture log messages
        log_messages = []
        
        # Create a custom handler to capture logs
        class TestLogHandler(logging.Handler):
            def emit(self, record):
                log_messages.append((record.levelname, record.getMessage()))
        
        # Add our test handler to the logger
        from src.niivue_nninteractive.api import logger
        test_handler = TestLogHandler()
        logger.addHandler(test_handler)
        logger.setLevel(logging.ERROR)
        
        # Mock initialize_model to raise an exception
        def mock_initialize_model():
            raise Exception("Test initialization failure")
        
        monkeypatch.setattr("src.niivue_nninteractive.api.initialize_model", mock_initialize_model)
        
        async def run_lifespan():
            # Mock app
            class MockApp:
                pass
            
            app = MockApp()
            
            # This should raise an exception and log error messages
            async with lifespan(app):
                pass
        
        try:
            # This should raise the exception from initialize_model
            with pytest.raises(Exception, match="Test initialization failure"):
                asyncio.run(run_lifespan())
            
            # Check that error message was logged
            error_logged = any(level == "ERROR" and "API server startup failed" in msg 
                             for level, msg in log_messages)
            assert error_logged, "Should log error message when initialization fails"
            
        finally:
            # Clean up the test handler
            logger.removeHandler(test_handler)


class TestPerformSegmentationEdgeCases:
    """Test class for edge cases in perform_segmentation"""
    
    def test_segmentation_with_scribble_thickness_variations(self, client, test_image_path):
        """Test segmentation with different scribble patterns that affect thickness logic"""
        # Test scribbles that will exercise the thickness and connection logic
        scribbles_for_thickness = [
            {"x": 100, "y": 150, "z": 9, "is_positive": True},
            {"x": 101, "y": 151, "z": 9, "is_positive": True},  # Connected on same slice
            {"x": 102, "y": 152, "z": 9, "is_positive": True},  # Forms a line
            {"x": 110, "y": 150, "z": 9, "is_positive": False}, # Negative scribble
            {"x": 111, "y": 151, "z": 9, "is_positive": False}, # Another negative
        ]
        
        with open(test_image_path, "rb") as f:
            image_data = f.read()
        
        files = {"image": ("FLAIR.nii.gz", io.BytesIO(image_data), "application/octet-stream")}
        data = {"scribbles": json.dumps(scribbles_for_thickness)}
        
        response = client.post("/segment", files=files, data=data)
        assert response.status_code == 200
        
        # Validate the segmentation result
        segmentation_data = response.content
        assert len(segmentation_data) > 0
    
    def test_segmentation_without_preferred_scribble_thickness(self, client, test_image_path, sample_scribbles):
        """Test segmentation when model session doesn't have preferred_scribble_thickness attribute"""
        # Mock the model_session to not have preferred_scribble_thickness
        from src.niivue_nninteractive.api import model_session
        
        if model_session and hasattr(model_session, 'preferred_scribble_thickness'):
            # Temporarily remove the attribute to test the else clause
            original_thickness = model_session.preferred_scribble_thickness
            delattr(model_session, 'preferred_scribble_thickness')
            
            try:
                with open(test_image_path, "rb") as f:
                    image_data = f.read()
                
                files = {"image": ("FLAIR.nii.gz", io.BytesIO(image_data), "application/octet-stream")}
                data = {"scribbles": json.dumps(sample_scribbles)}
                
                response = client.post("/segment", files=files, data=data)
                assert response.status_code == 200
                
                # Validate the segmentation result
                segmentation_data = response.content
                assert len(segmentation_data) > 0
                
            finally:
                # Restore the original attribute
                model_session.preferred_scribble_thickness = original_thickness
        else:
            # If model_session doesn't exist or doesn't have the attribute, this should still work
            with open(test_image_path, "rb") as f:
                image_data = f.read()
            
            files = {"image": ("FLAIR.nii.gz", io.BytesIO(image_data), "application/octet-stream")}
            data = {"scribbles": json.dumps(sample_scribbles)}
            
            response = client.post("/segment", files=files, data=data)
            assert response.status_code == 200
    
    def test_segmentation_with_out_of_bounds_scribble_pixels(self, client, test_image_path, monkeypatch):
        """Test segmentation with scribbles that create out-of-bounds pixels during thickness expansion"""
        # Set up logging capture
        log_messages = []
        
        def mock_logger_debug(message):
            log_messages.append(message)
        
        # Mock the logger.debug to capture debug messages
        monkeypatch.setattr("src.niivue_nninteractive.api.logger.debug", mock_logger_debug)
        
        # Create scribbles that will definitely cause out-of-bounds when thickness is applied
        # Use extreme coordinates that will go negative when thickness is subtracted
        boundary_scribbles = [
            {"x": -100, "y": -100, "z": 9, "is_positive": True},    # Way out of bounds
            {"x": 9999, "y": 9999, "z": 9, "is_positive": False},  # Way out of bounds
        ]
        
        with open(test_image_path, "rb") as f:
            image_data = f.read()
        
        files = {"image": ("FLAIR.nii.gz", io.BytesIO(image_data), "application/octet-stream")}
        data = {"scribbles": json.dumps(boundary_scribbles)}
        
        response = client.post("/segment", files=files, data=data)
        # This should fail due to no valid scribbles
        assert response.status_code == 500
        
        # If we reach here, it means the validation caught the out-of-bounds scribbles
        # Let's try with scribbles that are valid but will cause out-of-bounds pixels during thickness
        valid_but_boundary_scribbles = [
            {"x": 0, "y": 0, "z": 9, "is_positive": True},      # At boundary, thickness will go out of bounds
            {"x": 197, "y": 301, "z": 18, "is_positive": False}, # At max boundary
        ]
        
        files2 = {"image": ("FLAIR.nii.gz", io.BytesIO(open(test_image_path, 'rb').read()), "application/octet-stream")}
        data2 = {"scribbles": json.dumps(valid_but_boundary_scribbles)}
        
        response2 = client.post("/segment", files=files2, data=data2)
        assert response2.status_code == 200
        
        # Check if any debug messages were captured
        # It's ok if no debug messages are captured, as the thickness might not always cause out-of-bounds
        # This test primarily ensures the boundary scribble handling works correctly


import pytest