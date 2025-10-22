import os
import io
import uuid
import torch
import numpy as np
import SimpleITK as sitk
import logging
import sys
import traceback
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
from huggingface_hub import snapshot_download
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
import json
from datetime import datetime, timedelta
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup"""
    logger.info("Starting API server...")
    try:
        initialize_model()
        logger.info("API server startup completed successfully!")
    except Exception as e:
        logger.error(f"API server startup failed: {str(e)}")
        raise
    yield
    # Cleanup on shutdown if needed
    logger.info("API server shutting down...")

# Initialize FastAPI app
app = FastAPI(title="nnInteractive Segmentation API", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],  # Allow frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-User-ID"]  # Expose custom header
)

# Request/Response logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response

# Model configuration
REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"
DOWNLOAD_DIR = "./models"

# Session storage - in production, use Redis or similar
user_sessions: Dict[str, Dict] = {}
SESSION_TIMEOUT = timedelta(hours=1)

# Global model session (shared across users for efficiency)
model_session = None

class ScribbleCoordinate(BaseModel):
    x: int
    y: int
    z: int
    is_positive: bool = True


class SegmentationRequest(BaseModel):
    user_id: Optional[str] = None
    scribbles: List[ScribbleCoordinate]

def initialize_model():
    """Initialize the nnInteractive model session"""
    global model_session
    
    try:
        logger.info("Starting model initialization...")
        
        # Download model if not exists
        logger.info(f"Downloading model from {REPO_ID}...")
        download_path = snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=[f"{MODEL_NAME}/*"],
            local_dir=DOWNLOAD_DIR
        )
        logger.info(f"Model downloaded to: {download_path}")
        
        # Check device availability
        device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize session
        logger.info("Initializing nnInteractive session...")
        model_session = nnInteractiveInferenceSession(
            device=device,
            use_torch_compile=False,
            verbose=False,
            torch_n_threads=os.cpu_count(),
            do_autozoom=True,
            use_pinned_memory=True,
        )
        
        # Load model
        model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)
        logger.info(f"Loading model from: {model_path}")
        model_session.initialize_from_trained_model_folder(model_path)
        logger.info("Model initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


async def cleanup_old_sessions():
    """Remove expired sessions"""
    current_time = datetime.now()
    expired_users = []
    
    for user_id, session in user_sessions.items():
        if current_time - session['last_access'] > SESSION_TIMEOUT:
            expired_users.append(user_id)
    
    if expired_users:
        logger.info(f"Cleaning up {len(expired_users)} expired sessions")
        for user_id in expired_users:
            logger.info(f"Removing expired session: {user_id}")
            del user_sessions[user_id]

@app.post("/segment")
async def segment_image(
    image: Optional[UploadFile] = File(None),
    scribbles: str = Form(...),
    user_id: Optional[str] = Form(None)
):
    """
    Segment an image using scribble coordinates.
    
    First request must include image file.
    Subsequent requests can omit image to refine existing segmentation.
    """
    logger.info(f"Segmentation request received - user_id: {user_id}, has_image: {image is not None}")
    
    # Parse scribbles
    try:
        scribble_data = json.loads(scribbles)
        scribble_coords = [ScribbleCoordinate(**coord) for coord in scribble_data]
        logger.info(f"Parsed {len(scribble_coords)} scribbles")
        
        # Log scribble details
        positive_count = sum(1 for s in scribble_coords if s.is_positive)
        negative_count = len(scribble_coords) - positive_count
        logger.info(f"Scribbles: {positive_count} positive, {negative_count} negative")
        
        # Log raw scribble coordinates for debugging
        if scribble_coords:
            logger.info(f"Raw scribble coordinate samples: {scribble_coords[:3]}")
        
    except Exception as e:
        logger.error(f"Failed to parse scribbles: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid scribbles format: {str(e)}")
    
    # Generate or validate user_id
    if not user_id:
        user_id = str(uuid.uuid4())
    
    # Clean up old sessions
    await cleanup_old_sessions()
    
    # Check if user has existing session
    if user_id in user_sessions:
        session = user_sessions[user_id]
        session['last_access'] = datetime.now()
        
        if not image:
            # Use existing image
            img_array = session['image']
            input_image_sitk = session['image_sitk']
        else:
            # Update with new image
            img_array, input_image_sitk = await process_uploaded_image(image)
            session['image'] = img_array
            session['image_sitk'] = input_image_sitk
    else:
        # New session - image required
        if not image:
            raise HTTPException(
                status_code=400, 
                detail="Image file required for first request. Please provide a NIfTI image."
            )
        
        img_array, input_image_sitk = await process_uploaded_image(image)
        
        # Create new session
        user_sessions[user_id] = {
            'image': img_array,
            'image_sitk': input_image_sitk,
            'last_access': datetime.now()
        }
    
    # Perform segmentation
    try:
        logger.info(f"Starting segmentation for user {user_id}")
        logger.info(f"Image shape: {img_array.shape}")
        
        segmentation = perform_segmentation(img_array, scribble_coords)
        logger.info(f"Segmentation completed. Result shape: {segmentation.shape}")
        
        # Convert result to SimpleITK image
        results_sitk = sitk.GetImageFromArray(segmentation.cpu().numpy())
        results_sitk.CopyInformation(input_image_sitk)
        logger.info("Converted segmentation to SimpleITK format")
        
        # Save to temporary file then read to buffer
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
            tmp_path = tmp.name
        
        sitk.WriteImage(results_sitk, tmp_path)
        logger.info(f"Saved segmentation to temporary file: {tmp_path}")
        
        # Read the file into buffer
        with open(tmp_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        
        # Clean up temp file
        os.unlink(tmp_path)
        buffer.seek(0)
        
        logger.info(f"Segmentation request completed successfully for user {user_id}")
        
        return StreamingResponse(
            buffer,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename=segmentation.nii.gz",
                "X-User-ID": user_id
            }
        )
        
    except Exception as e:
        logger.error(f"Segmentation failed for user {user_id}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

async def process_uploaded_image(image: UploadFile) -> Tuple[np.ndarray, sitk.Image]:
    """Process uploaded NIfTI image"""
    logger.info(f"Processing uploaded image: {image.filename}")
    
    if not image.filename.endswith(('.nii', '.nii.gz')):
        logger.error(f"Invalid file type: {image.filename}")
        raise HTTPException(status_code=400, detail="Only NIfTI files (.nii, .nii.gz) are supported")
    
    # Read image data
    content = await image.read()
    logger.info(f"Read {len(content)} bytes from uploaded file")
    
    # Load with SimpleITK
    try:
        # Write to temporary file for SimpleITK
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info(f"Created temporary file: {tmp_path}")
        
        input_image = sitk.ReadImage(tmp_path)
        img_array = sitk.GetArrayFromImage(input_image)[None]  # Add batch dimension
        
        logger.info(f"Loaded image with shape: {img_array.shape}")
        logger.info(f"Image spacing: {input_image.GetSpacing()}")
        logger.info(f"Image origin: {input_image.GetOrigin()}")
        logger.info(f"Image direction: {input_image.GetDirection()}")
        logger.info(f"Image size: {input_image.GetSize()}")
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Validate dimensions
        if img_array.ndim != 4:
            logger.error(f"Invalid image dimensions: {img_array.ndim}")
            raise ValueError("Input image must be 3D (will be converted to 4D)")
        
        logger.info("Image processing completed successfully")
        return img_array, input_image
        
    except Exception as e:
        logger.error(f"Failed to process image: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Failed to load NIfTI image: {str(e)}")

def perform_segmentation(img_array: np.ndarray, scribbles: List[ScribbleCoordinate]) -> torch.Tensor:
    """Perform segmentation using scribble coordinates"""
    logger.info(f"Starting segmentation with {len(scribbles)} scribbles")
    logger.info(f"Input image array shape: {img_array.shape}")
    logger.info(f"Input image array dtype: {img_array.dtype}")
    
    # Log scribble coordinates for debugging
    for i, scribble in enumerate(scribbles[:5]):  # Log first 5 scribbles
        logger.info(f"Scribble {i}: x={scribble.x}, y={scribble.y}, z={scribble.z}, positive={scribble.is_positive}")
    
    # Set image in session
    logger.info("Setting image in model session")
    model_session.set_image(img_array)
    
    # Create target buffer
    target_tensor = torch.zeros(img_array.shape[1:], dtype=torch.uint8)
    model_session.set_target_buffer(target_tensor)
    logger.info(f"Created target buffer with shape: {target_tensor.shape}")
    
    # Reset any previous interactions
    logger.info("Resetting previous interactions")
    model_session.reset_interactions()
    
    # Validate scribble coordinates are within image bounds
    img_dims = img_array.shape[1:]  # Remove batch dimension - shape is [z, y, x]
    logger.info(f"Image dimensions for validation (z,y,x): {img_dims}")
    
    valid_scribbles = []
    for i, scribble in enumerate(scribbles):
        # Frontend sends (x,y,z) but array is (z,y,x), so check:
        # scribble.x against img_dims[2] (x dimension)
        # scribble.y against img_dims[1] (y dimension)
        # scribble.z against img_dims[0] (z dimension)
        if (0 <= scribble.x < img_dims[2] and 
            0 <= scribble.y < img_dims[1] and 
            0 <= scribble.z < img_dims[0]):
            valid_scribbles.append(scribble)
        else:
            logger.warning(f"Scribble {i} out of bounds: x={scribble.x}, y={scribble.y}, z={scribble.z}, bounds=(z={img_dims[0]}, y={img_dims[1]}, x={img_dims[2]})")
    
    logger.info(f"Valid scribbles: {len(valid_scribbles)}/{len(scribbles)}")
    
    if not valid_scribbles:
        raise ValueError("No valid scribbles found within image bounds")
    
    # Separate positive and negative scribbles
    positive_scribbles = [s for s in valid_scribbles if s.is_positive]
    negative_scribbles = [s for s in valid_scribbles if not s.is_positive]
    
    # Get scribble thickness
    if hasattr(model_session, 'preferred_scribble_thickness'):
        thickness = model_session.preferred_scribble_thickness
        scribble_thickness = thickness[0] if isinstance(thickness, list) else thickness
    else:
        scribble_thickness = 3
    
    # Create positive scribble image if we have positive scribbles
    if positive_scribbles:
        logger.info(f"Creating positive scribble image for {len(positive_scribbles)} scribbles")
        positive_scribble_image = np.zeros(img_array.shape[1:], dtype=np.uint8)
        
        # Create more substantial scribbles by connecting points
        for i, scribble in enumerate(positive_scribbles):
            # Draw thick points
            for dx in range(-scribble_thickness, scribble_thickness + 1):
                for dy in range(-scribble_thickness, scribble_thickness + 1):
                    x = scribble.x + dx
                    y = scribble.y + dy
                    z = scribble.z
                    
                    # Check bounds and index correctly (array is [z, y, x])
                    if (0 <= x < positive_scribble_image.shape[2] and 
                        0 <= y < positive_scribble_image.shape[1] and 
                        0 <= z < positive_scribble_image.shape[0]):
                        positive_scribble_image[z, y, x] = 1
                    else:
                        logger.debug(f"Scribble pixel out of bounds: x={x}, y={y}, z={z}, shape={positive_scribble_image.shape}")
            
            # Connect to next point if available
            if i < len(positive_scribbles) - 1:
                next_scribble = positive_scribbles[i + 1]
                # Draw line between points if on same slice
                if scribble.z == next_scribble.z:
                    steps = max(abs(next_scribble.x - scribble.x), abs(next_scribble.y - scribble.y))
                    if steps > 0:
                        for step in range(steps + 1):
                            t = step / steps
                            x = int(scribble.x + t * (next_scribble.x - scribble.x))
                            y = int(scribble.y + t * (next_scribble.y - scribble.y))
                            
                            # Apply thickness to line
                            for dx in range(-scribble_thickness//2, scribble_thickness//2 + 1):
                                for dy in range(-scribble_thickness//2, scribble_thickness//2 + 1):
                                    px = x + dx
                                    py = y + dy
                                    if (0 <= px < positive_scribble_image.shape[2] and 
                                        0 <= py < positive_scribble_image.shape[1] and
                                        0 <= scribble.z < positive_scribble_image.shape[0]):
                                        positive_scribble_image[scribble.z, py, px] = 1
        
        # Add positive scribble interaction
        logger.info(f"Adding positive scribble interaction - scribble pixels: {np.sum(positive_scribble_image)}")
        model_session.add_scribble_interaction(positive_scribble_image, include_interaction=True)
    
    # Create negative scribble image if we have negative scribbles
    if negative_scribbles:
        logger.info(f"Creating negative scribble image for {len(negative_scribbles)} scribbles")
        negative_scribble_image = np.zeros(img_array.shape[1:], dtype=np.uint8)
        
        for scribble in negative_scribbles:
            # Apply thickness around the point
            for dx in range(-scribble_thickness//2, scribble_thickness//2 + 1):
                for dy in range(-scribble_thickness//2, scribble_thickness//2 + 1):
                    x = scribble.x + dx
                    y = scribble.y + dy
                    z = scribble.z
                    
                    # Check bounds and index correctly (array is [z, y, x])
                    if (0 <= x < negative_scribble_image.shape[2] and 
                        0 <= y < negative_scribble_image.shape[1] and 
                        0 <= z < negative_scribble_image.shape[0]):
                        negative_scribble_image[z, y, x] = 1
                    else:
                        logger.debug(f"Negative scribble pixel out of bounds: x={x}, y={y}, z={z}, shape={negative_scribble_image.shape}")
        
        # Add negative scribble interaction with include_interaction=False
        logger.info(f"Adding negative scribble interaction - scribble pixels: {np.sum(negative_scribble_image)}")
        model_session.add_scribble_interaction(negative_scribble_image, include_interaction=False)
    
    # Return the segmentation result
    logger.info("Segmentation processing completed")
    result = target_tensor.clone()
    logger.info(f"Segmentation result - unique values: {torch.unique(result)}")
    return result

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    health_status = {
        "status": "healthy",
        "model_loaded": model_session is not None,
        "active_sessions": len(user_sessions)
    }
    logger.info(f"Health status: {health_status}")
    return health_status

@app.get("/")
async def root():
    """API information"""
    return {
        "api": "nnInteractive Segmentation API",
        "version": "1.0",
        "endpoints": {
            "/segment": "POST - Perform segmentation with NIfTI image and scribbles",
            "/health": "GET - Check API health status"
        },
        "usage": {
            "first_request": "Include NIfTI image file and scribble coordinates",
            "refinement": "Include user_id from first response and new scribble coordinates"
        }
    }