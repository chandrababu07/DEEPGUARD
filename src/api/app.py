"""
DeepGuard FastAPI application factory and API router.

Exposes:
- GET  /api/v1/health     : Service liveness probe
- GET  /api/v1/ready      : Model dependency and checkpoint readiness probe
- POST /api/v1/predict    : Upload video for deepfake prediction
- POST /api/v1/explain    : Upload video for Grad-CAM explainability heatmaps
- GET  /api/v1/artifacts/{filename} : Serve generated explanation images safely
- GET  /                  : Serve interactive web application UI
"""

import os
from pathlib import Path
import time
import uuid
from typing import Optional
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings, APISettings
from .image_service import default_image_service, ImageModelService
from .schemas import (
    ErrorResponse,
    ExplanationResponse,
    HealthResponse,
    ImagePredictionResponse,
    PredictionResponse,
    ReadinessResponse,
)
from .security import (
    resolve_safe_artifact_path,
    safe_cleanup,
    save_bounded_image_upload,
    save_bounded_upload,
    validate_decoded_image,
    validate_decoded_video,
)
from .service import default_model_service, ModelService


STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"


def create_app(
    model_service: Optional[ModelService] = None,
    image_service: Optional[ImageModelService] = None,
    custom_settings: Optional[APISettings] = None,
) -> FastAPI:
    """
    Factory creating configured FastAPI application.
    """
    app_settings = custom_settings or settings
    service = model_service or default_model_service
    img_service = image_service or (
        getattr(service, "image_service", None) or default_image_service
    )

    app = FastAPI(
        title="DeepGuard — Spatial-Temporal Deepfake Video Detection API",
        description=(
            "Secure, explainable computer-vision API for detecting deepfake videos "
            "using MobileNetV2 + LSTM temporal sequence analysis."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 1. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # 2. Request ID & Timing Middleware
    @app.middleware("http")
    async def request_id_and_timing_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = req_id
        start_time = time.perf_counter()

        try:
            response: Response = await call_next(request)
        except HTTPException as http_exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            resp = JSONResponse(
                status_code=http_exc.status_code,
                content=ErrorResponse(
                    error=http_exc.__class__.__name__,
                    detail=str(http_exc.detail),
                    request_id=req_id,
                    status_code=http_exc.status_code,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                ).model_dump(),
            )
            resp.headers["X-Request-ID"] = req_id
            resp.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
            return resp
        except Exception:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            resp = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="InternalServerError",
                    detail="An unexpected server error occurred while processing the request.",
                    request_id=req_id,
                    status_code=500,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                ).model_dump(),
            )
            resp.headers["X-Request-ID"] = req_id
            resp.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
            return resp

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response

    # 3. Global Exception Handlers (sanitizes internal errors, prevents leaking filesystem paths)
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        req_id = getattr(request.state, "request_id", uuid.uuid4().hex)
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.__class__.__name__,
                detail=str(exc.detail),
                request_id=req_id,
                status_code=exc.status_code,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        req_id = getattr(request.state, "request_id", uuid.uuid4().hex)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="InternalServerError",
                detail="An unexpected server error occurred while processing the request.",
                request_id=req_id,
                status_code=500,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ).model_dump(),
        )

    # 4. API Routes
    @app.get(
        "/api/v1/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Service Liveness Probe",
    )
    async def health_check():
        """Returns service liveness status."""
        return HealthResponse(
            status="healthy",
            service="DeepGuard Deepfake Detection API",
            version="1.0.0",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    @app.get(
        "/api/v1/ready",
        response_model=ReadinessResponse,
        tags=["System"],
        summary="Model & Inference Readiness Probe",
    )
    async def readiness_check():
        """Returns model readiness and dependency verification status."""
        return service.get_readiness()

    @app.post(
        "/api/v1/predict",
        response_model=PredictionResponse,
        tags=["Inference"],
        summary="Predict Video Manipulation",
    )
    async def predict_video(
        request: Request,
        file: UploadFile = File(..., description="Uploaded video file (.mp4, .avi, .mov, .webm)"),
    ):
        """
        Uploads and analyzes a video file for deepfake manipulation.
        """
        req_id = getattr(request.state, "request_id", uuid.uuid4().hex)
        temp_path = None

        try:
            # 1. Bounded streaming save with magic byte check
            temp_path, original_filename = await save_bounded_upload(
                upload_file=file,
                dest_dir=app_settings.upload_tmp_dir,
                max_bytes=app_settings.max_upload_size_bytes,
            )

            # 2. OpenCV stream validation
            validate_decoded_video(
                video_path=temp_path,
                max_duration_seconds=app_settings.max_video_duration_seconds,
                min_required_frames=app_settings.min_required_frames,
            )

            # 3. Model inference
            result = service.predict(
                video_path=temp_path,
                request_id=req_id,
                video_filename=original_filename,
            )
            return result

        finally:
            # Safe cleanup of uploaded temporary file
            safe_cleanup(temp_path)

    @app.post(
        "/api/v1/predict/image",
        response_model=ImagePredictionResponse,
        tags=["Inference"],
        summary="Predict Single-Image Manipulation",
    )
    async def predict_image(
        request: Request,
        file: UploadFile = File(..., description="Uploaded image file (.jpg, .jpeg, .png, .webp)"),
    ):
        """
        Uploads and analyzes a single image file for deepfake manipulation.
        Enforces size limits, magic bytes, decompression safety, and isolated image model inference.
        """
        req_id = getattr(request.state, "request_id", uuid.uuid4().hex)
        temp_path = None

        try:
            # 1. Bounded streaming save with image magic byte check
            temp_path, original_filename = await save_bounded_image_upload(
                upload_file=file,
                dest_dir=app_settings.upload_tmp_dir,
                max_bytes=app_settings.max_image_upload_size_bytes,
            )

            # 2. OpenCV decoding & spatial dimension validation
            validate_decoded_image(
                image_path=temp_path,
                max_dimension=app_settings.max_image_dimension,
            )

            # 3. Dedicated image model inference
            result = img_service.predict(
                image_path=temp_path,
                request_id=req_id,
                image_filename=original_filename,
            )
            return result

        finally:
            # Safe cleanup of uploaded temporary image file
            safe_cleanup(temp_path)

    @app.post(
        "/api/v1/explain",
        response_model=ExplanationResponse,
        tags=["Explainability"],
        summary="Generate Grad-CAM Spatial Attribution Heatmaps",
    )
    async def explain_video(
        request: Request,
        file: UploadFile = File(..., description="Uploaded video file (.mp4, .avi, .mov, .webm)"),
    ):
        """
        Uploads a video and generates Grad-CAM spatial attribution heatmap overlays.
        """
        req_id = getattr(request.state, "request_id", uuid.uuid4().hex)
        temp_path = None

        try:
            temp_path, original_filename = await save_bounded_upload(
                upload_file=file,
                dest_dir=app_settings.upload_tmp_dir,
                max_bytes=app_settings.max_upload_size_bytes,
            )

            validate_decoded_video(
                video_path=temp_path,
                max_duration_seconds=app_settings.max_video_duration_seconds,
                min_required_frames=app_settings.min_required_frames,
            )

            explanation = service.explain(
                video_path=temp_path,
                request_id=req_id,
                video_filename=original_filename,
                artifact_dir=app_settings.artifact_storage_dir,
                base_artifact_url="/api/v1/artifacts",
            )
            return explanation

        finally:
            safe_cleanup(temp_path)

    @app.get(
        "/api/v1/artifacts/{artifact_filename}",
        tags=["Explainability"],
        summary="Retrieve Explanation Heatmap Artifact",
    )
    async def get_artifact(artifact_filename: str):
        """
        Safely serve an explanation overlay image with path traversal protection.
        """
        safe_path = resolve_safe_artifact_path(
            artifact_filename=artifact_filename,
            base_dir=app_settings.artifact_storage_dir,
        )
        return FileResponse(safe_path)

    # 5. Frontend Integration
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            index_path = STATIC_DIR / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return JSONResponse({"message": "DeepGuard API is running. UI index.html not found."})

    return app


# Default app instance
app = create_app()
