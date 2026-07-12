"""Stage 4a — Landmarks. Per-character geometry derived from part silhouettes.

Implementation lives in :mod:`.extract`; this package re-exports its public API.
"""

from __future__ import annotations

from .extract import (
    AlphaSampler,
    analyze_silhouette,
    BrowLandmarks,
    DEFAULT_ALPHA_THRESHOLD,
    DEFAULT_SAMPLES,
    detect_face_landmarks_ml,
    detect_pose_ml,
    extract_landmarks,
    EyeLandmarks,
    FaceLandmarkDetector,
    landmark_warnings,
    Landmarks,
    landmarks_from_silhouettes,
    MouthLandmarks,
    Oval,
    PoseDetector,
    render_overlay,
    Silhouette,
)

__all__ = [
    "AlphaSampler",
    "analyze_silhouette",
    "BrowLandmarks",
    "DEFAULT_ALPHA_THRESHOLD",
    "DEFAULT_SAMPLES",
    "detect_face_landmarks_ml",
    "detect_pose_ml",
    "extract_landmarks",
    "EyeLandmarks",
    "FaceLandmarkDetector",
    "landmark_warnings",
    "Landmarks",
    "landmarks_from_silhouettes",
    "MouthLandmarks",
    "Oval",
    "PoseDetector",
    "render_overlay",
    "Silhouette",
]
