#!/usr/bin/env python3
"""
Enhanced Vision Controller for PAROL6 Robot
============================================
Handles all vision-related processing with improved reliability:
- RealSense camera initialization and management
- 2D to 3D coordinate transformation with validation
- Advanced depth extraction and filtering
- Hand-eye calibration application
- Reachability and safety analysis
- Temporal filtering for stability
- Debug visualization support
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import time
import math
import json
from typing import List, Tuple, Optional, Dict, Any, Deque
from dataclasses import dataclass, field
from collections import deque
from spatialmath import SE3
import threading

from Headless.robot_api import (
    get_robot_pose, 
    get_robot_pose_matrix,
    move_robot_pose, 
    jog_robot_joint
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Camera Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
DEPTH_FORMAT = rs.format.z16
COLOR_FORMAT = rs.format.bgr8

# Calibration
DEFAULT_CALIBRATION_FILE = "Headless\Vision\Results\calibration\calibration_manual_20250828_174636.npz"

# Robot Configuration
ROBOT_MAX_REACH_MM = 400
ROBOT_MIN_REACH_MM = 100  # Minimum safe distance
ROBOT_WORKSPACE_HEIGHT_MAX = 300  # Maximum height
ROBOT_WORKSPACE_HEIGHT_MIN = -50  # Minimum height (below base)

# Depth Processing
DEPTH_FILTER_SIZE = 5  # Median filter kernel size
MIN_VALID_DEPTH_RATIO = 0.2  # Lowered from 0.3 for small objects
MAX_VALID_DEPTH_MM = 2000  # Maximum valid depth reading
MIN_VALID_DEPTH_MM = 100  # Minimum valid depth reading
DEPTH_TEMPORAL_WINDOW = 5  # Number of frames for temporal filtering
DEPTH_OUTLIER_THRESHOLD = 30  # Reduced from 50mm for tighter filtering

# Safety Parameters
MIN_GRIPPER_CLEARANCE = 50  # mm clearance needed around target
APPROACH_ANGLE_THRESHOLD = 45  # Maximum approach angle from vertical

# Visualization
ENABLE_DEBUG_VISUALIZATION = True
DEBUG_WINDOW_NAME = "Vision Controller Debug"

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Detection3D:
    """Enhanced 3D detection result with confidence metrics"""
    bbox_2d: List[int]  # [x1, y1, x2, y2]
    position_3d: np.ndarray  # [x, y, z] in robot base frame (mm)
    position_camera: np.ndarray  # [x, y, z] in camera frame (mm)
    depth_mm: float
    depth_variance: float  # Variance in depth readings
    confidence: float  # Overall confidence score
    depth_confidence: float  # Confidence in depth measurement
    is_reachable: bool
    reachability_score: float  # 0-1 score of how reachable
    distance_from_base: float
    angle_to_center: float  # Angle to rotate base to center object
    suggested_approach: Dict[str, Any] = field(default_factory=dict)
    safety_clearance: bool = True
    timestamp: float = field(default_factory=time.time)
    # NEW FIELDS:
    depth_stats: Optional[Dict[str, float]] = None  # Comprehensive depth statistics
    object_orientation: float = 0.0  # Object rotation angle in degrees

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    
    @classmethod
    def from_rs_intrinsics(cls, intrinsics):
        """Create from RealSense intrinsics"""
        return cls(
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.ppx,
            cy=intrinsics.ppy,
            width=intrinsics.width,
            height=intrinsics.height
        )

# ============================================================================
# VISION CONTROLLER
# ============================================================================

class VisionController:
    """Enhanced vision processing controller with improved reliability"""
    
    def __init__(self, calibration_file: str = DEFAULT_CALIBRATION_FILE):
        """Initialize vision controller with calibration data"""
        # Camera components
        self.pipeline = None
        self.align = None
        self.intrinsics = None
        self.camera_intrinsics = None
        
        # Calibration data
        self.calibration_data = None
        self.load_calibration(calibration_file)
        
        # Temporal filtering
        self.depth_history = deque(maxlen=DEPTH_TEMPORAL_WINDOW)
        self.detection_history = deque(maxlen=3)
        
        # Thread safety
        self.frame_lock = threading.Lock()
        self.latest_color_frame = None
        self.latest_depth_frame = None
        
        # Statistics
        self.frame_count = 0
        self.error_count = 0
        
        # Debug visualization
        self.debug_mode = ENABLE_DEBUG_VISUALIZATION
        if self.debug_mode:
            cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    
    def load_calibration(self, filepath: str):
        """Load and validate hand-eye calibration data"""
        try:
            data = np.load(filepath)
            
            # Validate required fields
            required_fields = ['camera_matrix', 'dist_coeffs', 'T_cam2gripper']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            self.calibration_data = {
                'camera_matrix': data['camera_matrix'].astype(np.float64),
                'dist_coeffs': data['dist_coeffs'].astype(np.float64),
                'T_cam2gripper': data['T_cam2gripper'].astype(np.float64)
            }
            
            # Validate transformation matrix
            if self.calibration_data['T_cam2gripper'].shape != (4, 4):
                raise ValueError("Invalid transformation matrix shape")
            
            # Transformation matrix validation
            det = np.linalg.det(self.calibration_data['T_cam2gripper'][:3, :3])
            if abs(det - 1.0) > 0.1:
                print(f"⚠ Warning: Transformation matrix determinant = {det:.3f}")
            
            print(f"✓ Loaded calibration from {filepath}")
            
            # Extract and display camera offset
            offset = self.calibration_data['T_cam2gripper'][:3, 3] * 1000
            print(f"  Camera offset: X:{offset[0]:.1f}mm, Y:{offset[1]:.1f}mm, Z:{offset[2]:.1f}mm")
            
            # Store additional calibration info if available
            if 'rms_error' in data:
                print(f"  Calibration RMS error: {data['rms_error']:.3f}")
            
        except FileNotFoundError:
            print(f"⚠ Calibration file not found: {filepath}")
            print("  Using default calibration values")
            self._use_default_calibration()
            
        except Exception as e:
            print(f"✗ Error loading calibration: {e}")
            print("  Using default calibration values")
            self._use_default_calibration()
    
    def _use_default_calibration(self):
        """Use default calibration values as fallback"""
        self.calibration_data = {
            'camera_matrix': np.array([
                [615.0, 0, 320.0],
                [0, 615.0, 240.0],
                [0, 0, 1]
            ], dtype=np.float64),
            'dist_coeffs': np.zeros(5, dtype=np.float64),
            'T_cam2gripper': np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0.057],  # 57mm Y offset
                [0, 0, 1, -0.049],  # 49mm Z offset
                [0, 0, 0, 1]
            ], dtype=np.float64)
        }
    
    def initialize_camera(self) -> bool:
        """Initialize RealSense camera with optimal settings"""
        try:
            # RealSense pipeline initialization
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Camera stream configuration
            config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, DEPTH_FORMAT, CAMERA_FPS)
            config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, COLOR_FORMAT, CAMERA_FPS)
            
            # Start pipeline
            profile = self.pipeline.start(config)
            
            # Get device for advanced settings
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            
            # Enable advanced settings if available
            if depth_sensor.supports(rs.option.enable_auto_exposure):
                depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
            
            # Set depth units to mm
            if depth_sensor.supports(rs.option.depth_units):
                depth_sensor.set_option(rs.option.depth_units, 0.001)
            
            # Get intrinsics
            color_stream = profile.get_stream(rs.stream.color)
            self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            self.camera_intrinsics = CameraIntrinsics.from_rs_intrinsics(self.intrinsics)
            
            # Depth-color alignment setup
            self.align = rs.align(rs.stream.color)
            
            # Apply post-processing filters
            self.spatial_filter = rs.spatial_filter()
            self.temporal_filter = rs.temporal_filter()
            self.hole_filling = rs.hole_filling_filter()
            
            # Depth filtering configuration
            self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
            
            # Let auto-exposure settle
            print("Initializing camera (auto-exposure settling)...")
            for i in range(30):
                self.pipeline.wait_for_frames()
                if i % 10 == 0:
                    print(f"  {30-i} frames remaining...")
            
            print("✓ Camera initialized successfully")
            print(f"  Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
            print(f"  FOV: H={self._calculate_fov_h():.1f}°, V={self._calculate_fov_v():.1f}°")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize camera: {e}")
            return False
    
    def _calculate_fov_h(self) -> float:
        """Calculate horizontal field of view"""
        return 2 * math.degrees(math.atan(self.camera_intrinsics.width / (2 * self.camera_intrinsics.fx)))
    
    def _calculate_fov_v(self) -> float:
        """Calculate vertical field of view"""
        return 2 * math.degrees(math.atan(self.camera_intrinsics.height / (2 * self.camera_intrinsics.fy)))
    
    def stop_camera(self):
        """Stop camera pipeline and cleanup"""
        if self.pipeline:
            self.pipeline.stop()
            print("✓ Camera stopped")
        
        if self.debug_mode:
            cv2.destroyWindow(DEBUG_WINDOW_NAME)
    
    def get_frames(self, apply_filters: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get aligned and filtered color and depth frames
        
        Args:
            apply_filters: Whether to apply post-processing filters
            
        Returns:
            Tuple of (color_frame, depth_frame) as numpy arrays
        """
        if not self.pipeline:
            return None, None
        
        try:
            # Get frameset
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            # Align frames
            aligned_frames = self.align.process(frames)
            
            # Get individual frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                self.error_count += 1
                return None, None
            
            # Apply post-processing filters to depth
            if apply_filters:
                depth_frame = self.spatial_filter.process(depth_frame)
                depth_frame = self.temporal_filter.process(depth_frame)
                depth_frame = self.hole_filling.process(depth_frame)
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Update latest frames (thread-safe)
            with self.frame_lock:
                self.latest_color_frame = color_image.copy()
                self.latest_depth_frame = depth_image.copy()
                self.frame_count += 1
            
            return color_image, depth_image
            
        except Exception as e:
            self.error_count += 1
            if self.error_count % 10 == 0:  # Log every 10th error
                print(f"Error getting frames ({self.error_count} total): {e}")
            return None, None
        
    def extract_depth_statistics(self, 
                        bbox: List[int], 
                        depth_frame: np.ndarray,
                        robot_pose: np.ndarray,
                        verbose: bool = True) -> Dict[str, float]:
        """
        Extract depth statistics in world frame using point cloud transformation.
        Now with improved ROI extraction and diagnostic logging.
        
        Returns:
            Dictionary with:
            - object_top: Highest Z in world frame (actual top of object)
            - object_median: Median Z in world frame
            - table_surface: Lowest Z in world frame (actual table)
            - object_height: True height in mm
            - confidence: Reliability score
            - world_centroid: [x, y, z] true centroid in world frame
            - centroid_variance: [var_x, var_y, var_z] variance of points
            - centroid_reliable: Whether centroid can be trusted
            - valid_point_ratio: Ratio of valid depth points (diagnostic)
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure bbox is within frame bounds
        h, w = depth_frame.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        # FIX 1: Increase ROI from 60% to 85% for better coverage of small objects
        center_ratio = 0.85  # Was 0.6
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rx, ry = int((x2 - x1) * center_ratio / 2), int((y2 - y1) * center_ratio / 2)
        
        roi_x1, roi_x2 = max(0, cx - rx), min(w, cx + rx)
        roi_y1, roi_y2 = max(0, cy - ry), min(h, cy + ry)
        
        # Pixel coordinate mesh grid
        xx, yy = np.meshgrid(np.arange(roi_x1, roi_x2), 
                            np.arange(roi_y1, roi_y2))
        
        # Get depth values for ROI
        depth_roi = depth_frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Filter valid depths
        valid_mask = (depth_roi > MIN_VALID_DEPTH_MM) & (depth_roi < MAX_VALID_DEPTH_MM)
        
        # Point cloud quality metrics
        valid_point_ratio = np.sum(valid_mask) / valid_mask.size if valid_mask.size > 0 else 0
        
        if verbose:
            print(f"[DEPTH] ROI size: {roi_x2-roi_x1}x{roi_y2-roi_y1}, Valid points: {np.sum(valid_mask)}/{valid_mask.size} ({valid_point_ratio*100:.1f}%)")
        
        if np.sum(valid_mask) < 10:  # Need minimum samples
            if verbose:
                print(f"[DEPTH] WARNING: Too few valid points ({np.sum(valid_mask)})")
            return None
        
        # Get valid pixel coordinates and depths
        valid_x = xx[valid_mask].flatten()
        valid_y = yy[valid_mask].flatten()
        valid_depths = depth_roi[valid_mask].flatten()
        
        # Transform all valid points to world frame
        world_points = []
        
        for px, py, depth_mm in zip(valid_x, valid_y, valid_depths):
            # Deproject to 3D camera coordinates
            point_camera = rs.rs2_deproject_pixel_to_point(
                self.intrinsics,
                [px, py],
                depth_mm
            )
            point_camera = np.array(point_camera) / 1000.0  # Convert to meters
            
            # Transform: camera -> gripper -> world
            point_camera_h = np.append(point_camera, 1.0)
            point_gripper_h = self.calibration_data['T_cam2gripper'] @ point_camera_h
            point_world_h = robot_pose @ point_gripper_h
            
            # Extract world coordinates in mm
            world_point = point_world_h[:3] * 1000
            world_points.append(world_point)
        
        world_points = np.array(world_points)
        
        # Extract Z coordinates (height in world frame)
        world_z = world_points[:, 2]
        
        # FIX 1: Use 10th/90th percentiles instead of 5th/95th for more robust measurements
        z_min = np.percentile(world_z, 10)   # Table surface (10th percentile)
        z_max = np.percentile(world_z, 90)   # Object top (90th percentile)
        z_median = np.median(world_z)
        
        # Object height estimation
        object_height = z_max - z_min
        
        if verbose:
            print(f"[DEPTH] World Z range: {z_min:.1f} to {z_max:.1f}mm, Height: {object_height:.1f}mm, Median: {z_median:.1f}mm")
        
        # World coordinate centroid calculation
        world_centroid = np.mean(world_points, axis=0)
        centroid_variance = np.var(world_points, axis=0)
        
        # FIX 1: Better confidence calculation based on multiple factors
        # Consider: valid point ratio, height consistency, and variance
        height_confidence = min(1.0, object_height / 25.0) if object_height > 0 else 0
        point_confidence = valid_point_ratio
        variance_confidence = 1.0 / (1.0 + np.mean(centroid_variance) / 100.0)
        
        # Combined confidence
        confidence = (height_confidence * 0.4 + point_confidence * 0.4 + variance_confidence * 0.2)
        
        # Centroid reliability assessment
        # More lenient criteria: need decent points and reasonable variance
        centroid_reliable = (valid_point_ratio > 0.2 and  # At least 20% valid points
                            object_height > 5 and          # At least 5mm height detected
                            np.mean(centroid_variance) < 500)  # Reasonable variance
        
        if verbose:
            print(f"[DEPTH] Confidence: {confidence:.2f} (height:{height_confidence:.2f}, points:{point_confidence:.2f}, var:{variance_confidence:.2f})")
            print(f"[DEPTH] Centroid: [{world_centroid[0]:.1f}, {world_centroid[1]:.1f}, {world_centroid[2]:.1f}], Reliable: {centroid_reliable}")
        
        return {
            'object_top': z_max,
            'object_median': z_median,
            'table_surface': z_min,
            'object_height': object_height,
            'confidence': confidence,
            'world_centroid': world_centroid,
            'centroid_variance': centroid_variance,
            'centroid_reliable': centroid_reliable,
            'valid_point_ratio': valid_point_ratio,  # Diagnostic
            'world_points': world_points  # Keep for orientation calculation
        }
    
    def calculate_object_orientation_world_frame(self, world_points: np.ndarray) -> float:
        """
        Calculate object orientation using PCA on world-frame points.
        DISABLED for small objects to prevent mis-rotation.
        """
        # FIX: Disable rotation for small objects (height < 30mm)
        if 'depth_stats' in locals() or hasattr(self, '_current_depth_stats'):
            # Get height from current processing context
            object_height = self._current_depth_stats.get('object_height', 0) if hasattr(self, '_current_depth_stats') else 0
            
            if object_height < 30:
                print(f"[ROTATION] Disabled for small object (height={object_height:.1f}mm)")
                return 0.0
        
        # Also check point count
        if len(world_points) < 100:
            print(f"[ROTATION] Insufficient points for reliable rotation ({len(world_points)} points)")
            return 0.0
        
        # Project points onto world XY plane (remove Z)
        xy_points = world_points[:, :2]
        
        # Apply PCA in world frame
        mean = np.mean(xy_points, axis=0)
        centered = xy_points - mean
        cov_matrix = np.cov(centered.T)
        
        # Handle degenerate cases
        if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
            return 0.0
        
        try:
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            
            # Primary axis is eigenvector with largest eigenvalue
            primary_axis = eigenvectors[:, np.argmax(eigenvalues)]
            
            # Calculate angle in world frame
            angle = np.arctan2(primary_axis[1], primary_axis[0])
            angle_degrees = np.degrees(angle)
            
            # Normalize to [-90, 90]
            while angle_degrees > 90:
                angle_degrees -= 180
            while angle_degrees < -90:
                angle_degrees += 180
            
            return float(angle_degrees)
            
        except np.linalg.LinAlgError:
            return 0.0
    
    def extract_depth_from_bbox(self, 
                               bbox: List[int], 
                               depth_frame: np.ndarray,
                               use_temporal: bool = True) -> Tuple[Optional[float], float]:
        """
        Extract robust depth measurement with confidence from bounding box
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            depth_frame: Depth image as numpy array
            use_temporal: Whether to use temporal filtering
            
        Returns:
            Tuple of (depth_mm, variance) or (None, 0) if failed
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure bbox is within frame bounds
        h, w = depth_frame.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        # Validate bbox size
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            print("⚠ Bounding box too small for reliable depth")
            return None, 0
        
        # Extract center region (more reliable than edges)
        center_ratio = 0.6  # Use central 60% of bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rx, ry = int((x2 - x1) * center_ratio / 2), int((y2 - y1) * center_ratio / 2)
        
        # Extract ROI from center
        roi_x1, roi_x2 = max(0, cx - rx), min(w, cx + rx)
        roi_y1, roi_y2 = max(0, cy - ry), min(h, cy + ry)
        depth_roi = depth_frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Filter out invalid readings
        valid_depths = depth_roi[
            (depth_roi > MIN_VALID_DEPTH_MM) & 
            (depth_roi < MAX_VALID_DEPTH_MM)
        ]
        
        if len(valid_depths) == 0:
            return None, 0
        
        # Check validity ratio
        valid_ratio = len(valid_depths) / depth_roi.size
        if valid_ratio < MIN_VALID_DEPTH_RATIO:
            print(f"⚠ Only {valid_ratio*100:.1f}% valid depth readings")
            return None, 0
        
        # Remove outliers using IQR method
        q1, q3 = np.percentile(valid_depths, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_depths = valid_depths[
            (valid_depths >= lower_bound) & 
            (valid_depths <= upper_bound)
        ]
        
        if len(filtered_depths) == 0:
            filtered_depths = valid_depths  # Fall back if all filtered out
        
        # Calculate statistics
        depth_mm = float(np.median(filtered_depths))
        variance = float(np.var(filtered_depths))
        
        # Apply temporal filtering if enabled
        if use_temporal:
            self.depth_history.append(depth_mm)
            if len(self.depth_history) > 1:
                # Weighted average with recent frames
                weights = np.exp(np.linspace(-1, 0, len(self.depth_history)))
                weights /= weights.sum()
                depth_mm = float(np.average(list(self.depth_history), weights=weights))
        
        return depth_mm, variance
    
    def bbox_to_3d_position(self, 
                  bbox: List[int], 
                  depth_frame: np.ndarray,
                  robot_pose: Optional[np.ndarray] = None,
                  validate_safety: bool = True) -> Optional[Detection3D]:
        """
        Enhanced version using world-frame point cloud processing with centroid correction.
        """
        # Calculate bbox center
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Get robot pose if not provided
        if robot_pose is None:
            robot_pose = get_robot_pose_matrix()
            if robot_pose is None:
                print("✗ Failed to get robot pose")
                return None
        
        # Extract world-frame depth statistics (includes centroid now)
        depth_stats = self.extract_depth_statistics(bbox, depth_frame, robot_pose)
        if depth_stats is None:
            print("✗ Failed to extract depth statistics in world frame")
            return None
        
        # Get the depth at center pixel for initial transformation
        center_depth = depth_frame[center_y, center_x]
        if center_depth < MIN_VALID_DEPTH_MM or center_depth > MAX_VALID_DEPTH_MM:
            # If center pixel is invalid, use median from ROI
            roi = depth_frame[y1:y2, x1:x2]
            valid_depths = roi[(roi > MIN_VALID_DEPTH_MM) & (roi < MAX_VALID_DEPTH_MM)]
            if len(valid_depths) > 0:
                center_depth = float(np.median(valid_depths))
            else:
                return None
        
        # Deproject center to 3D camera coordinates (original method)
        point_camera = rs.rs2_deproject_pixel_to_point(
            self.intrinsics,
            [center_x, center_y],
            center_depth
        )
        point_camera = np.array(point_camera)
        
        # Transform: camera -> gripper -> robot base
        point_camera_h = np.append(point_camera / 1000.0, 1.0)  # Convert to meters
        point_gripper_h = self.calibration_data['T_cam2gripper'] @ point_camera_h
        point_base_h = robot_pose @ point_gripper_h
        
        # Extract 3D position in mm (this is the bbox-center derived position)
        position_3d_original = point_base_h[:3] * 1000  # Convert back to mm
        
        # FIX 2: Enhanced centroid correction with Y-offset compensation
        if depth_stats.get('centroid_reliable', False) and depth_stats['confidence'] > 0.3:  # Lowered from 0.5
            world_centroid = depth_stats['world_centroid']
            
            # Calculate correction offsets
            x_correction = world_centroid[0] - position_3d_original[0]
            y_correction = world_centroid[1] - position_3d_original[1]
            
            # Apply limits to corrections (max 30mm adjustment)
            MAX_CORRECTION = 30.0  # mm
            x_correction = np.clip(x_correction, -MAX_CORRECTION, MAX_CORRECTION)
            y_correction = np.clip(y_correction, -MAX_CORRECTION, MAX_CORRECTION)
            
            # Check if corrections are reasonable
            correction_magnitude = np.sqrt(x_correction**2 + y_correction**2)
            
            if correction_magnitude < MAX_CORRECTION:
                # Apply corrections with confidence-based weighting
                weight = depth_stats['confidence']
                position_3d = position_3d_original.copy()
                position_3d[0] += x_correction * weight
                position_3d[1] += y_correction * weight
                position_3d[2] = depth_stats['object_median']  # Always use median for Z
                
                if self.debug_mode or validate_safety:  # Add verbose flag
                    print(f"[3D] Applied centroid correction: X={x_correction*weight:.1f}mm, Y={y_correction*weight:.1f}mm (confidence={weight:.2f})")
            else:
                # Correction too large, fall back to original with just Z fix
                print(f"[3D] Centroid correction too large ({correction_magnitude:.1f}mm), using fallback")
                position_3d = position_3d_original.copy()
                position_3d[2] = depth_stats['object_median']
        else:
            # FIX 2: When centroid not reliable, apply systematic Y-offset compensation
            # Camera is mounted 39mm behind gripper, causing systematic forward bias
            print(f"[3D] Centroid not reliable (conf={depth_stats['confidence']:.2f}), applying Y-offset compensation")
            position_3d = position_3d_original.copy()
            position_3d[2] = depth_stats['object_median']
            
            # Apply systematic Y-offset correction for camera mounting geometry
            # Camera sees front edge as center, need to shift back
            Y_OFFSET_COMPENSATION = -12.0  # mm, negative because camera is behind gripper
            position_3d[1] += Y_OFFSET_COMPENSATION
            
            if self.debug_mode:
                print(f"[3D] Applied Y-offset compensation: {Y_OFFSET_COMPENSATION}mm")
                print(f"[3D] Final position: X={position_3d[0]:.1f}, Y={position_3d[1]:.1f}, Z={position_3d[2]:.1f}")
        
        # Calculate derived metrics
        distance_from_base = float(np.linalg.norm(position_3d[:2]))  # XY distance
        height = position_3d[2]
        
        # Enhanced reachability analysis
        is_reachable, reachability_score = self._analyze_reachability(
            position_3d, distance_from_base, height
        )
        
        # Calculate angle to center object
        angle_to_center = math.degrees(math.atan2(position_3d[1], position_3d[0]))
        
        # Calculate confidence
        depth_confidence = depth_stats['confidence']
        overall_confidence = depth_confidence

        # Store depth stats temporarily for rotation calculation
        self._current_depth_stats = depth_stats
        
        # Calculate object orientation in world frame
        object_orientation = 0.0
        if 'world_points' in depth_stats and depth_stats['object_height'] > 30:  # Only for objects > 30mm
            object_orientation = self.calculate_object_orientation_world_frame(
                depth_stats['world_points']
            )
        else:
            print(f"[ROTATION] Skipped: height={depth_stats.get('object_height', 0):.1f}mm")
        
        # Clear temporary storage
        self._current_depth_stats = None
        
        # Determine suggested approach
        suggested_approach = self._calculate_approach_strategy(position_3d, height)
        
        # Safety validation
        safety_clearance = True
        if validate_safety:
            safety_clearance = self._validate_safety_clearance(position_3d)
        
        # Create detection object with world-frame statistics
        detection = Detection3D(
            bbox_2d=bbox,
            position_3d=position_3d,
            position_camera=point_camera,
            depth_mm=center_depth,
            depth_variance=0.0,  # Could use centroid_variance if needed
            confidence=overall_confidence,
            depth_confidence=depth_confidence,
            is_reachable=is_reachable,
            reachability_score=reachability_score,
            distance_from_base=distance_from_base,
            angle_to_center=angle_to_center,
            suggested_approach=suggested_approach,
            safety_clearance=safety_clearance,
            depth_stats=depth_stats,  # Now includes centroid data
            object_orientation=object_orientation
        )
        
        # Add to history for temporal consistency
        self.detection_history.append(detection)
        
        # Debug visualization if enabled
        if self.debug_mode:
            self._visualize_detection(detection, depth_frame)
        
        return detection
    
    def _calculate_depth_confidence(self, variance: float, depth: float) -> float:
        """Calculate confidence score for depth measurement"""
        # Lower variance = higher confidence
        # Normalize by depth (farther objects naturally have more variance)
        normalized_variance = variance / depth if depth > 0 else 1.0
        
        # Convert to 0-1 confidence score
        confidence = 1.0 / (1.0 + normalized_variance * 0.01)
        return min(1.0, max(0.0, confidence))
    
    def _analyze_reachability(self, 
                            position: np.ndarray, 
                            distance: float, 
                            height: float) -> Tuple[bool, float]:
        """
        Analyze if position is reachable with a confidence score
        
        Returns:
            Tuple of (is_reachable, reachability_score)
        """
        # Basic reachability check
        in_range = ROBOT_MIN_REACH_MM <= distance <= ROBOT_MAX_REACH_MM
        in_height = ROBOT_WORKSPACE_HEIGHT_MIN <= height <= ROBOT_WORKSPACE_HEIGHT_MAX
        
        is_reachable = in_range and in_height
        
        # Calculate reachability score (0-1)
        if not is_reachable:
            return False, 0.0
        
        # Distance score (prefer middle of range)
        optimal_distance = (ROBOT_MAX_REACH_MM + ROBOT_MIN_REACH_MM) / 2
        distance_score = 1.0 - abs(distance - optimal_distance) / optimal_distance
        
        # Height score (prefer middle heights)
        optimal_height = (ROBOT_WORKSPACE_HEIGHT_MAX + ROBOT_WORKSPACE_HEIGHT_MIN) / 2
        height_score = 1.0 - abs(height - optimal_height) / ROBOT_WORKSPACE_HEIGHT_MAX
        
        # Combined score
        reachability_score = (distance_score * 0.6 + height_score * 0.4)
        
        return is_reachable, max(0.0, min(1.0, reachability_score))
    
    def _calculate_approach_strategy(self, position: np.ndarray, height: float) -> Dict[str, Any]:
        """Calculate optimal approach strategy for grasping"""
        strategy = {
            'angle': 0,  # Default vertical
            'speed': 30,  # Default speed percentage
            'clearance': MIN_GRIPPER_CLEARANCE,
            'orientation': [0, 180, 0]  # Default gripper orientation
        }
        
        # Adjust based on height
        if height > 200:
            strategy['angle'] = 0  # Vertical approach from above
            strategy['orientation'] = [0, 180, 0]
        elif height > 100:
            strategy['angle'] = 45  # Angled approach
            strategy['orientation'] = [45, 180, 0]
        else:
            strategy['angle'] = 90  # Horizontal approach
            strategy['orientation'] = [90, 180, 0]
        
        # Adjust speed based on distance
        distance = np.linalg.norm(position[:2])
        if distance < 200:
            strategy['speed'] = 20  # Slower for close objects
        elif distance > 350:
            strategy['speed'] = 40  # Faster for far objects
        
        return strategy
    
    def _validate_safety_clearance(self, position: np.ndarray) -> bool:
        """Validate if there's enough clearance for safe operation"""
        # Check minimum clearance from base
        xy_distance = np.linalg.norm(position[:2])
        if xy_distance < ROBOT_MIN_REACH_MM - MIN_GRIPPER_CLEARANCE:
            return False
        
        # Check height clearance
        if position[2] < ROBOT_WORKSPACE_HEIGHT_MIN + MIN_GRIPPER_CLEARANCE:
            return False
        
        return True
    
    def calculate_rotation_to_center(self, 
                                    bbox: List[int], 
                                    tolerance_pixels: int = 20) -> Optional[float]:
        """
        Calculate rotation needed to center object in camera view
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            tolerance_pixels: Dead zone in pixels
            
        Returns:
            Rotation angle in degrees, or None if already centered
        """
        # Calculate bbox center
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        
        # Frame center
        frame_center_x = self.camera_intrinsics.cx
        
        # Calculate pixel error
        error_pixels = center_x - frame_center_x
        
        if abs(error_pixels) < tolerance_pixels:
            return None  # Already centered
        
        # Convert to angle using camera FOV
        fov_horizontal = self._calculate_fov_h()
        angle_per_pixel = fov_horizontal / self.camera_intrinsics.width
        rotation_needed = error_pixels * angle_per_pixel
        
        return rotation_needed
    
    def _visualize_detection(self, detection: Detection3D, depth_frame: np.ndarray):
        """Visualize detection for debugging"""
        if self.latest_color_frame is None:
            return
        
        # Create visualization
        vis_frame = self.latest_color_frame.copy()
        
        # Draw bounding box
        x1, y1, x2, y2 = detection.bbox_2d
        color = (0, 255, 0) if detection.is_reachable else (0, 0, 255)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Add text info
        info_text = [
            f"Depth: {detection.depth_mm:.0f}mm",
            f"Conf: {detection.confidence:.2f}",
            f"Reach: {'Yes' if detection.is_reachable else 'No'} ({detection.reachability_score:.2f})",
            f"Angle: {detection.angle_to_center:.1f}°"
        ]
        
        y_offset = y1 - 10
        for text in info_text:
            cv2.putText(vis_frame, text, (x1, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset -= 20
        
        # Show depth colormap alongside
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_frame, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        # Combine both views
        combined = np.hstack((vis_frame, depth_colormap))
        cv2.imshow(DEBUG_WINDOW_NAME, combined)
        cv2.waitKey(1)
    
    def get_camera_stats(self) -> Dict[str, Any]:
        """Get camera and processing statistics"""
        return {
            'frame_count': self.frame_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.frame_count),
            'depth_history_size': len(self.depth_history),
            'detection_history_size': len(self.detection_history),
            'fps': CAMERA_FPS,
            'resolution': f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"
        }
    
    def start_display_thread(self, window_name: str = "Camera Preview"):
        """
        Start a thread that continuously displays the camera feed
        
        Args:
            window_name: Name for the preview window
        """
        self.display_window_name = window_name
        self.display_running = True
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        print(f"✓ Started camera preview in '{window_name}' window")

    def _display_loop(self):
        """
        Internal method that runs in a separate thread to display camera feed
        """
        cv2.namedWindow(self.display_window_name, cv2.WINDOW_AUTOSIZE)
        
        while self.display_running:
            try:
                # Get the latest frames
                with self.frame_lock:
                    if self.latest_color_frame is not None:
                        display_frame = self.latest_color_frame.copy()
                    else:
                        # Create a gray placeholder if no frame available
                        display_frame = np.ones((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8) * 128
                        cv2.putText(display_frame, "Waiting for camera...", (50, CAMERA_HEIGHT//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add overlay information
                if self.latest_color_frame is not None:
                    # Add frame counter
                    cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add timestamp
                    cv2.putText(display_frame, f"FPS: {CAMERA_FPS}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add status
                    if self.error_count > 0:
                        cv2.putText(display_frame, f"Errors: {self.error_count}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display the frame
                cv2.imshow(self.display_window_name, display_frame)
                
                # Check for 'q' key to close window (optional)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    print("Preview window closed by user")
                    break
                    
            except Exception as e:
                print(f"Display error: {e}")
                # Create error frame
                display_frame = np.ones((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8) * 64
                cv2.putText(display_frame, "Display Error", (50, CAMERA_HEIGHT//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(self.display_window_name, display_frame)
                cv2.waitKey(100)
        
        # Cleanup
        cv2.destroyWindow(self.display_window_name)
        print(f"Camera preview window closed")

    def stop_display_thread(self):
        """
        Stop the display thread and close the preview window
        """
        if hasattr(self, 'display_running'):
            self.display_running = False
            if hasattr(self, 'display_thread'):
                self.display_thread.join(timeout=1.0)
            print("✓ Stopped camera preview")

    # Also modify the stop_camera method in VisionController to include:
    def stop_camera(self):
        """Stop camera pipeline and cleanup"""
        # Stop display thread if running
        if hasattr(self, 'display_running') and self.display_running:
            self.stop_display_thread()
        
        if self.pipeline:
            self.pipeline.stop()
            print("✓ Camera stopped")
        
        if self.debug_mode:
            cv2.destroyWindow(DEBUG_WINDOW_NAME)

# ============================================================================
# HELPER FUNCTIONS (for backward compatibility)
# ============================================================================

# Global controller instance
_vision_controller = None

def initialize_realsense() -> rs.pipeline:
    """Initialize RealSense camera (backward compatibility)"""
    global _vision_controller
    _vision_controller = VisionController()
    
    if _vision_controller.initialize_camera():
        return _vision_controller.pipeline
    return None

def get_vision_controller() -> VisionController:
    """Get or create global vision controller instance"""
    global _vision_controller
    if _vision_controller is None:
        _vision_controller = VisionController()
        _vision_controller.initialize_camera()
    return _vision_controller

def process_bounding_box(bbox: List[int]) -> Optional[Detection3D]:
    """
    Process a bounding box and return 3D detection
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box
        
    Returns:
        Detection3D object or None
    """
    controller = get_vision_controller()
    
    # Get frames
    color_frame, depth_frame = controller.get_frames()
    if color_frame is None or depth_frame is None:
        return None
    
    # Convert to 3D
    return controller.bbox_to_3d_position(bbox, depth_frame)

def enable_debug_visualization(enable: bool = True):
    """Enable or disable debug visualization"""
    controller = get_vision_controller()
    controller.debug_mode = enable
    if enable and controller.debug_mode:
        cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    elif not enable:
        cv2.destroyWindow(DEBUG_WINDOW_NAME)