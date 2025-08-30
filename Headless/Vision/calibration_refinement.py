#!/usr/bin/env python3
"""
Improved Hand-Eye Calibration Refinement for PAROL6 Robot
Fixes critical bugs and implements comprehensive improvements:
- Class-based architecture with persistent state tracking
- Robust board localization using all 15 positions with multiple samples
- Hemisphere sampling with FOV awareness and rotation diversity
- Live visualization window
- Mixed exploration/refinement strategy
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import time
import socket
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
import json
from collections import defaultdict
import os
from datetime import datetime
import random

# Robot API imports
from Headless.robot_api import (
    move_robot_joints,
    move_robot_cartesian,
    get_robot_pose,
    get_robot_pose_matrix,
    get_robot_joint_speeds,
    get_robot_joint_angles,
    stop_robot_movement,
    delay_robot
)

print(f"OpenCV Version: {cv2.__version__}")

# ChArUco board parameters
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH = 0.025  # 25mm in meters
MARKER_LENGTH = 0.015  # 15mm in meters
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
BOARD_CHANGE_WARNING_THRESHOLD = 0.02  # 20mm

# Known working positions from original calibration
KNOWN_WORKING_POSITIONS = [
    [16.26, -83.725, 152.689, -47.461, -78.117, 241.712],
    [30.779, -85.168, 139.991, -43.481, -64.336, 209.346],
    [30.779, -85.168, 139.991, -34.566, -81.098, 103.523],
    [30.779, -85.168, 139.991, -34.566, -64.308, 180.366],
    [47.971, -85.168, 139.991, -62.142, -64.308, 269.381],
    [47.971, -85.168, 139.991, -42.483, -59.358, 149.468],
    [68.616, -85.168, 139.991, -55.28, -33.666, 228.403],
    [68.616, -103.238, 153.022, -41.47, -57.361, 250.616],
    [100.934, -91.527, 153.479, 25.65, -46.547, 143.696],
    [121.192, -91.527, 153.479, 38.475, -63.464, 93.178],
    [121.192, -91.527, 153.479, 42.272, -53.48, 129.611],
    [121.192, -48.752, 205.093, 25.369, -76.092, 162.039],
    [90.0, -59.102, 197.66, -20.728, -73.195, 220.59],
    [68.792, -59.102, 174.483, -45.436, -63.309, 261.096],
    [62.719, -93.308, 145.294, -50.273, -54.422, 261.872]
]

# Safe positions
SAFE_START_POSITION = [84.709, -87.491, 168.742, -5.006, -63.633, 182.002]

# Configuration
MAX_ITERATIONS = 3
CONVERGENCE_THRESHOLD = 0.5  # mm - research shows this is sufficient
SAMPLES_PER_POSITION = 5  # For board localization
MIN_ROTATION_DIVERSITY = 30  # degrees
MANIPULABILITY_THRESHOLD = 0.1  # Avoid singularities


@dataclass
class CalibrationData:
    """Stores calibration data point"""
    R_gripper2base: np.ndarray
    t_gripper2base: np.ndarray
    R_target2cam: np.ndarray
    t_target2cam: np.ndarray
    joint_angles: List[float]
    board_distance: float
    corners_detected: int


class RealSenseCamera:
    """RealSense D435I camera management"""
    
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        self.pipeline.start(self.config)
        
        print("Initializing camera...")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        time.sleep(2)
        
    def capture_image(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to capture color frame")
        return np.asanyarray(color_frame.get_data())
    
    def get_intrinsics(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        
        # CRITICAL: Ensure float64 for OpenCV
        camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)
        
        return camera_matrix, dist_coeffs
    
    def close(self):
        self.pipeline.stop()


class ImprovedCalibrationRefinement:
    """Main calibration refinement class with persistent state"""
    
    def __init__(self):
        # Persistent state across iterations
        self.pose_history = []  # ALL poses ever tried
        self.successful_poses = []  # Poses with good detections
        self.calibration_results = []  # Track convergence
        self.board_location = None  # Stable board position in base frame
        self.board_std = None  # Standard deviation of board location
        
        # NEW: Track iteration metrics
        self.iteration_metrics = []  # Performance per iteration
        self.board_location_history = []  # Track board position changes
        self.iteration_weights = []  # Track weights used

        # Phase 2: Spatial memory
        self.detection_memory = {}  # (grid_x, grid_y, grid_z) -> {'successes': int, 'attempts': int, 'avg_corners': float}
        self.grid_resolution = 0.02  # 20mm grid cells
        
        # Phase 3: Kalman-like state estimation
        self.board_position_estimate = None  # Current best estimate
        self.board_position_covariance = None  # Uncertainty matrix (3x3)
        self.process_noise = np.eye(3) * 1e-6  # Board doesn't move (very small noise)
        self.base_measurement_noise = np.eye(3) * 0.001  # 1mm base uncertainty
        
        # Camera and calibration data
        self.camera = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.T_cam2gripper_current = None
        
        # Visualization
        self.viz_window = "Calibration Live View"
        cv2.namedWindow(self.viz_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.viz_window, 800, 600)
        
        # IK validation
        self.ik_socket = None
        
        # Track data collection
        self.all_collected_data = []
        
    def load_existing_calibration(self) -> bool:
        """Load existing calibration results"""
        try:
            data = np.load("Results/calibration/calibration_results_improved.npz")
        except:
            try:
                data = np.load("calibration_results.npz")
                print("Using original calibration file")
            except:
                print("ERROR: No calibration file found!")
                return False
        
        # CRITICAL: Ensure float64
        self.camera_matrix = data['camera_matrix'].astype(np.float64)
        self.dist_coeffs = data['dist_coeffs'].astype(np.float64)
        self.T_cam2gripper_current = data['T_cam2gripper'].astype(np.float64)
        
        current_trans = self.T_cam2gripper_current[:3, 3] * 1000
        print(f"Current calibration: X={current_trans[0]:.1f}mm, "
              f"Y={current_trans[1]:.1f}mm, Z={current_trans[2]:.1f}mm")
        
        return True
    
    # PHASE 1: New quality-based weight calculation
    def calculate_detection_weight(self, corners_detected: int, board_distance: float) -> float:
        """
        Calculate weight based on detection quality, not iteration number.
        Higher corner count = quadratically higher weight
        Optimal distance around 200-250mm = higher weight
        """
        # Maximum possible corners is 24 (interior corners of 7x5 board)
        max_corners = 24
        
        # Quadratic weighting for corner count (more corners = much better)
        corner_weight = (corners_detected / max_corners) ** 2
        
        # Distance weight - penalize too close or too far
        # Optimal range is 200-250mm based on your data
        optimal_distance = 225  # mm
        distance_std = 50  # mm
        distance_weight = np.exp(-((board_distance - optimal_distance) / distance_std) ** 2)
        
        # Combined weight
        total_weight = corner_weight * distance_weight
        
        # Never let weight go to exactly zero
        return max(total_weight, 0.01)
    
    # PHASE 2: Spatial memory tracking
    def update_detection_memory(self, pose_matrix: np.ndarray, corners_detected: int, success: bool):
        """
        Track detection success patterns in 3D space.
        """
        # Get grid cell for this position
        position = pose_matrix[:3, 3]
        grid_cell = tuple(
            int(np.round(position[i] / self.grid_resolution))
            for i in range(3)
        )
        
        # Initialize if new cell
        if grid_cell not in self.detection_memory:
            self.detection_memory[grid_cell] = {
                'successes': 0,
                'attempts': 0,
                'avg_corners': 0.0,
                'positions': []
            }
        
        # Update statistics
        cell_data = self.detection_memory[grid_cell]
        cell_data['attempts'] += 1
        if success and corners_detected >= 6:
            cell_data['successes'] += 1
            # Running average of corners detected
            alpha = 0.3  # Learning rate
            cell_data['avg_corners'] = (1 - alpha) * cell_data['avg_corners'] + alpha * corners_detected
        cell_data['positions'].append(position.tolist())
    
    def get_detection_probability(self, pose_matrix: np.ndarray) -> float:
        """
        Estimate probability of successful detection at a given pose based on memory.
        """
        position = pose_matrix[:3, 3]
        
        # Check nearby grid cells (3x3x3 neighborhood)
        total_weight = 0
        weighted_probability = 0
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    grid_cell = (
                        int(np.round(position[0] / self.grid_resolution)) + dx,
                        int(np.round(position[1] / self.grid_resolution)) + dy,
                        int(np.round(position[2] / self.grid_resolution)) + dz
                    )
                    
                    if grid_cell in self.detection_memory:
                        cell_data = self.detection_memory[grid_cell]
                        if cell_data['attempts'] > 0:
                            # Weight by inverse distance
                            distance = np.sqrt(dx**2 + dy**2 + dz**2) + 0.5
                            weight = 1.0 / distance
                            
                            success_rate = cell_data['successes'] / cell_data['attempts']
                            weighted_probability += weight * success_rate
                            total_weight += weight
        
        if total_weight > 0:
            return weighted_probability / total_weight
        else:
            # No history - assume moderate probability
            return 0.5
    
    def detect_board_and_estimate_pose(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """Detect ChArUco board and estimate its pose using new API"""
        # Create board (no longer cv2.aruco.CharucoBoard directly)
        board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y), 
            SQUARE_LENGTH, 
            MARKER_LENGTH, 
            ARUCO_DICT
        )
        board.setLegacyPattern(True)
        
        # Create detector (new API)
        detector = cv2.aruco.CharucoDetector(board)
        
        # Detect board (returns 4 values in new API)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        
        if charuco_corners is None or len(charuco_corners) < 6:
            return None, None, 0
        
        # Get object points for detected corners - ENSURE float64
        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
        
        if obj_points is None or len(obj_points) < 4:
            return None, None, len(charuco_corners)
        
        # CRITICAL: Ensure float64 for solvePnP
        obj_points = obj_points.astype(np.float64)
        img_points = img_points.astype(np.float64)
        
        # Use solvePnP to estimate pose
        success, rvec, tvec = cv2.solvePnP(
            obj_points, 
            img_points, 
            self.camera_matrix.astype(np.float64), 
            self.dist_coeffs.astype(np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # Convert to 4x4 transformation matrix
            R, _ = cv2.Rodrigues(rvec.astype(np.float64))
            T_board2cam = np.eye(4, dtype=np.float64)
            T_board2cam[:3, :3] = R
            T_board2cam[:3, 3] = tvec.flatten()
            
            # Draw for visualization (updated function name and namespace)
            debug_img = image.copy()
            cv2.aruco.drawDetectedCornersCharuco(debug_img, charuco_corners, charuco_ids)
            cv2.drawFrameAxes(debug_img, self.camera_matrix, self.dist_coeffs, 
                            rvec, tvec, 0.05)
            
            return T_board2cam, debug_img, len(charuco_corners)
        
        return None, None, len(charuco_corners)
    
    def establish_robust_board_location(self):
        """Establish board location using ALL 15 positions with multiple samples"""
        print("\n=== Establishing Robust Board Location ===")
        print(f"Using all {len(KNOWN_WORKING_POSITIONS)} known positions")
        print(f"Taking {SAMPLES_PER_POSITION} samples per position")
        
        all_detections = []
        
        for i, position in enumerate(KNOWN_WORKING_POSITIONS):
            print(f"\n[{i+1}/{len(KNOWN_WORKING_POSITIONS)}] Moving to position {i+1}...")
            result = move_robot_joints(position, speed_percentage=20, wait_for_ack=True, timeout=15.0)
            
            if not self.wait_for_movement_completion(result, settle_time=1.5):
                print("  [WARNING] Movement incomplete, skipping")
                continue
            
            # Get robot pose once
            T_gripper2base = get_robot_pose_matrix()
            if T_gripper2base is None:
                print("  [ERROR] Failed to get robot pose")
                continue
            T_gripper2base = T_gripper2base.astype(np.float64)
            
            # Collect multiple samples at this position
            position_detections = []
            for sample in range(SAMPLES_PER_POSITION):
                time.sleep(0.1)  # Brief pause between samples
                
                image = self.camera.capture_image()
                T_board2cam, debug_img, corners = self.detect_board_and_estimate_pose(image)
                
                if T_board2cam is not None:
                    # Calculate board position in base frame
                    T_board2gripper = self.T_cam2gripper_current @ T_board2cam
                    T_board2base = T_gripper2base @ T_board2gripper
                    
                    position_detections.append({
                        'position': T_board2base[:3, 3],
                        'corners': corners,
                        'full_transform': T_board2base
                    })
                    
                # Update visualization
                if debug_img is not None:
                    self.update_visualization(debug_img, corners, 
                                            f"Board Localization [{i+1}/{len(KNOWN_WORKING_POSITIONS)}]")
            
            if len(position_detections) >= 3:
                # Use median for this position
                positions = np.array([d['position'] for d in position_detections])
                median_pos = np.median(positions, axis=0)
                std_pos = np.std(positions, axis=0) * 1000  # Convert to mm
                
                all_detections.append({
                    'position': median_pos,
                    'std_mm': std_pos,
                    'num_samples': len(position_detections),
                    'avg_corners': np.mean([d['corners'] for d in position_detections])
                })
                
                print(f"  [SUCCESS] Board detected: X={median_pos[0]*1000:.1f}mm, "
                      f"Y={median_pos[1]*1000:.1f}mm, Z={median_pos[2]*1000:.1f}mm")
                print(f"    Std Dev: X={std_pos[0]:.1f}mm, Y={std_pos[1]:.1f}mm, Z={std_pos[2]:.1f}mm")
            else:
                print(f"  [ERROR] Insufficient detections ({len(position_detections)}/{SAMPLES_PER_POSITION})")
        
        if len(all_detections) < 5:
            print("\nERROR: Too few positions detected board reliably")
            return False
        
        # Apply RANSAC for outlier removal
        positions = np.array([d['position'] for d in all_detections])
        inliers = self.ransac_filter(positions, threshold=0.005)  # 5mm threshold
        
        # Weighted average by detection quality
        weights = np.array([d['avg_corners'] * d['num_samples'] for i, d in enumerate(all_detections) if i in inliers])
        weights = weights / weights.sum()
        
        inlier_positions = positions[inliers]
        self.board_location = np.average(inlier_positions, weights=weights, axis=0)
        self.board_std = np.std(inlier_positions, axis=0) * 1000  # Convert to mm
        
        print(f"\nBoard location established ({len(inliers)}/{len(all_detections)} inliers):")
        print(f"  Position: X={self.board_location[0]*1000:.1f}mm, "
              f"Y={self.board_location[1]*1000:.1f}mm, Z={self.board_location[2]*1000:.1f}mm")
        print(f"  Std Dev:  X={self.board_std[0]:.1f}mm, Y={self.board_std[1]:.1f}mm, Z={self.board_std[2]:.1f}mm")
        
        return True
    
    def ransac_filter(self, positions: np.ndarray, threshold: float = 0.005, min_inliers: int = 5) -> List[int]:
        """Simple RANSAC to filter outlier positions"""
        best_inliers = []
        n_iterations = 100
        
        for _ in range(n_iterations):
            # Random sample
            sample_idx = np.random.choice(len(positions), min(3, len(positions)), replace=False)
            sample_mean = positions[sample_idx].mean(axis=0)
            
            # Find inliers
            distances = np.linalg.norm(positions - sample_mean, axis=1)
            inliers = np.where(distances < threshold)[0]
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers.tolist()
        
        return best_inliers if len(best_inliers) >= min_inliers else list(range(len(positions)))
    
    def generate_diverse_poses(self, iteration: int, num_poses: int = 100) -> List[Dict]:
        """
        Generate poses using spatial memory to bias toward successful regions.
        70% exploitation of known-good areas, 30% exploration of new areas.
        Maintains the original's proper spherical coordinate generation.
        """
        poses = []
        
        if self.board_location is None:
            print("Warning: No board location, using default")
            board_x, board_y, board_z = 0.032, 0.153, 0.001
        else:
            board_x = self.board_location[0]
            board_y = self.board_location[1] 
            board_z = self.board_location[2]
        
        print(f"\nGenerating poses with spatial memory (iteration {iteration})")
        print(f"  Board target: X={board_x*1000:.1f}mm, Y={board_y*1000:.1f}mm, Z={board_z*1000:.1f}mm")
        
        # Get the inverse of camera-to-gripper transform
        try:
            T_gripper2cam = np.linalg.inv(self.T_cam2gripper_current)
        except:
            print("Warning: Could not invert T_cam2gripper, using identity")
            T_gripper2cam = np.eye(4)
        
        # Determine exploitation vs exploration ratio based on available memory
        n_exploit = 0
        n_explore = num_poses
        
        if len(self.detection_memory) > 0 and iteration > 0:
            exploitation_ratio = 0.7
            exploration_ratio = 0.3
            n_exploit = int(num_poses * exploitation_ratio)
            n_explore = num_poses - n_exploit
            
            print(f"  Using spatial memory: {n_exploit} exploitation, {n_explore} exploration poses")
            
            # EXPLOITATION: Generate poses near successful detections
            successful_cells = [
                (cell, data) for cell, data in self.detection_memory.items()
                if data['successes'] > 0 and data['avg_corners'] > 8  # Only use reasonably good cells
            ]
            
            if successful_cells:
                # Sort by success rate * average corners (combined metric)
                successful_cells.sort(
                    key=lambda x: (x[1]['successes'] / x[1]['attempts']) * x[1]['avg_corners'],
                    reverse=True
                )
                
                print(f"    Found {len(successful_cells)} successful grid cells to exploit")
                
                poses_generated = 0
                attempts = 0
                max_attempts = n_exploit * 3  # Prevent infinite loop
                
                while poses_generated < n_exploit and attempts < max_attempts:
                    attempts += 1
                    
                    # Pick from top successful cells with exponential distribution
                    # This biases toward best cells but still explores variety
                    cell_idx = min(
                        int(np.random.exponential(scale=1.5)),
                        len(successful_cells) - 1
                    )
                    cell, cell_data = successful_cells[cell_idx]
                    
                    # Convert grid cell back to approximate position
                    base_position = np.array(cell) * self.grid_resolution
                    
                    # Add controlled random perturbation
                    # Perturbation is smaller for very successful cells
                    success_rate = cell_data['successes'] / cell_data['attempts']
                    perturbation_scale = self.grid_resolution * (1.5 - success_rate)  # 0.5-1.5 grid cells
                    perturbation = np.random.randn(3) * perturbation_scale
                    
                    camera_pos = base_position + perturbation
                    
                    # Add small rotation variations for diversity
                    roll_var = np.random.uniform(-10, 10) * np.pi / 180
                    pitch_var = np.random.uniform(-15, 15) * np.pi / 180
                    
                    # Create look-at matrix for camera
                    board_pos = np.array([board_x, board_y, board_z])
                    
                    z_axis = (board_pos - camera_pos)
                    if np.linalg.norm(z_axis) < 0.01:  # Too close to board center
                        continue
                    z_axis = z_axis / np.linalg.norm(z_axis)
                    
                    up = np.array([0, 0, 1])
                    x_axis = np.cross(up, z_axis)
                    if np.linalg.norm(x_axis) < 0.001:
                        up = np.array([1, 0, 0])
                        x_axis = np.cross(up, z_axis)
                    x_axis = x_axis / np.linalg.norm(x_axis)
                    y_axis = np.cross(z_axis, x_axis)
                    
                    # Apply rotation variations
                    R_base = np.column_stack([x_axis, y_axis, z_axis])
                    R_roll = np.array([
                        [1, 0, 0],
                        [0, np.cos(roll_var), -np.sin(roll_var)],
                        [0, np.sin(roll_var), np.cos(roll_var)]
                    ])
                    R_pitch = np.array([
                        [np.cos(pitch_var), 0, np.sin(pitch_var)],
                        [0, 1, 0],
                        [-np.sin(pitch_var), 0, np.cos(pitch_var)]
                    ])
                    R_final = R_base @ R_pitch @ R_roll
                    
                    T_camera = np.eye(4)
                    T_camera[:3, :3] = R_final
                    T_camera[:3, 3] = camera_pos
                    
                    # Transform to gripper pose
                    T_gripper = T_camera @ T_gripper2cam
                    
                    poses.append({
                        'pose_matrix': T_gripper,
                        'type': 'exploit_memory',
                        'params': {
                            'camera_distance': np.linalg.norm(camera_pos - board_pos),
                            'base_cell': cell,
                            'success_rate': success_rate,
                            'avg_corners': cell_data['avg_corners']
                        }
                    })
                    poses_generated += 1
                
                # Update how many exploration poses we need
                actual_exploit = poses_generated
                n_explore = num_poses - actual_exploit
                print(f"    Generated {actual_exploit} exploitation poses")
            else:
                print("    No successful cells found, using all exploration")
                n_explore = num_poses
        else:
            print(f"  No spatial memory available, using {n_explore} exploration poses")
        
        # EXPLORATION: Generate new poses to discover good regions
        # Using the original's proven strategies with proper spherical coordinates
        
        # Strategy 1: Hemisphere sampling (proper spherical coordinates)
        n_hemisphere = min(n_explore // 3, 40)
        if n_hemisphere > 0:
            angles_azimuth = np.random.uniform(0, 2*np.pi, n_hemisphere)
            angles_elevation = np.random.uniform(10, 60, n_hemisphere) * np.pi / 180
            distances = np.random.uniform(0.15, 0.35, n_hemisphere)
            
            for i in range(n_hemisphere):
                # CORRECT spherical coordinates (from original)
                cam_x = board_x + distances[i] * np.sin(angles_elevation[i]) * np.cos(angles_azimuth[i])
                cam_y = board_y + distances[i] * np.sin(angles_elevation[i]) * np.sin(angles_azimuth[i])
                cam_z = board_z + distances[i] * np.cos(angles_elevation[i])
                
                camera_pos = np.array([cam_x, cam_y, cam_z])
                board_pos = np.array([board_x, board_y, board_z])
                
                # Camera looks at board
                z_axis = (board_pos - camera_pos)
                z_axis = z_axis / np.linalg.norm(z_axis)
                
                up = np.array([0, 0, 1])
                x_axis = np.cross(up, z_axis)
                if np.linalg.norm(x_axis) < 0.001:
                    up = np.array([1, 0, 0])
                    x_axis = np.cross(up, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)
                
                T_camera = np.eye(4)
                T_camera[:3, 0] = x_axis
                T_camera[:3, 1] = y_axis
                T_camera[:3, 2] = z_axis
                T_camera[:3, 3] = camera_pos
                
                T_gripper = T_camera @ T_gripper2cam
                
                poses.append({
                    'pose_matrix': T_gripper,
                    'type': 'hemisphere_explore',
                    'params': {
                        'camera_distance': distances[i],
                        'azimuth': angles_azimuth[i] * 180 / np.pi,
                        'elevation': angles_elevation[i] * 180 / np.pi
                    }
                })
        
        # Strategy 2: Grid with random perturbations
        n_grid = min(n_explore // 3, 40)
        if n_grid > 0 and len(poses) < num_poses:
            x_offsets = np.random.uniform(-0.04, 0.04, n_grid)
            y_offsets = np.random.uniform(-0.03, 0.03, n_grid)
            z_heights = np.random.uniform(0.18, 0.28, n_grid)
            
            roll_variations = np.random.uniform(-15, 15, n_grid) * np.pi / 180
            pitch_variations = np.random.uniform(-20, 20, n_grid) * np.pi / 180
            
            for i in range(min(n_grid, num_poses - len(poses))):
                cam_x = board_x + x_offsets[i]
                cam_y = board_y + y_offsets[i]
                cam_z = board_z + z_heights[i]
                
                camera_pos = np.array([cam_x, cam_y, cam_z])
                board_pos = np.array([board_x, board_y, board_z])
                
                z_axis = (board_pos - camera_pos)
                z_axis = z_axis / np.linalg.norm(z_axis)
                
                up = np.array([0, 0, 1])
                x_axis = np.cross(up, z_axis)
                if np.linalg.norm(x_axis) < 0.001:
                    up = np.array([1, 0, 0])
                    x_axis = np.cross(up, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)
                
                # Apply rotational variations
                R_base = np.column_stack([x_axis, y_axis, z_axis])
                R_roll = np.array([
                    [1, 0, 0],
                    [0, np.cos(roll_variations[i]), -np.sin(roll_variations[i])],
                    [0, np.sin(roll_variations[i]), np.cos(roll_variations[i])]
                ])
                R_pitch = np.array([
                    [np.cos(pitch_variations[i]), 0, np.sin(pitch_variations[i])],
                    [0, 1, 0],
                    [-np.sin(pitch_variations[i]), 0, np.cos(pitch_variations[i])]
                ])
                R_final = R_base @ R_pitch @ R_roll
                
                T_camera = np.eye(4)
                T_camera[:3, :3] = R_final
                T_camera[:3, 3] = camera_pos
                
                T_gripper = T_camera @ T_gripper2cam
                
                poses.append({
                    'pose_matrix': T_gripper,
                    'type': 'grid_explore',
                    'params': {
                        'x_offset': x_offsets[i],
                        'y_offset': y_offsets[i],
                        'height': z_heights[i],
                        'roll': roll_variations[i] * 180 / np.pi,
                        'pitch': pitch_variations[i] * 180 / np.pi
                    }
                })
        
        # Strategy 3: Orbit around board
        n_orbit = min(num_poses - len(poses), 20)
        if n_orbit > 0:
            for i in range(n_orbit):
                angle = np.random.uniform(0, 2*np.pi)
                radius = np.random.uniform(0.12, 0.25)
                height = np.random.uniform(0.15, 0.30)
                
                # CORRECT circular orbit (from original fix)
                cam_x = board_x + radius * np.cos(angle)
                cam_y = board_y + radius * np.sin(angle)
                cam_z = board_z + height
                
                camera_pos = np.array([cam_x, cam_y, cam_z])
                board_pos = np.array([board_x, board_y, board_z])
                
                z_axis = (board_pos - camera_pos)
                z_axis = z_axis / np.linalg.norm(z_axis)
                
                up = np.array([0, 0, 1])
                x_axis = np.cross(up, z_axis)
                if np.linalg.norm(x_axis) < 0.001:
                    up = np.array([1, 0, 0])
                    x_axis = np.cross(up, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)
                
                T_camera = np.eye(4)
                T_camera[:3, 0] = x_axis
                T_camera[:3, 1] = y_axis
                T_camera[:3, 2] = z_axis
                T_camera[:3, 3] = camera_pos
                
                T_gripper = T_camera @ T_gripper2cam
                
                poses.append({
                    'pose_matrix': T_gripper,
                    'type': 'orbit_explore',
                    'params': {
                        'angle': angle * 180 / np.pi,
                        'radius': radius,
                        'height': height
                    }
                })
        
        # Shuffle poses for better diversity in selection
        random.shuffle(poses)
        
        # Log statistics
        pose_types = {}
        for pose in poses:
            pose_types[pose['type']] = pose_types.get(pose['type'], 0) + 1
        
        print(f"  Generated {len(poses)} total poses:")
        for ptype, count in pose_types.items():
            print(f"    - {ptype}: {count}")
        
        return poses
    
    def generate_hemisphere_poses(self, num_poses: int) -> List[Dict]:
        """Generate poses on hemisphere around board with FOV awareness"""
        poses = []
        board_center = self.board_location  # This is in meters: [0.033, 0.153, 0.0005]
        
        # Parameters for hemisphere sampling
        # Distance from board center (not radius of hemisphere)
        distances = np.linspace(0.15, 0.30, 4)  # 150-300mm from board center
        # Height above board
        heights = np.linspace(0.10, 0.25, 4)  # 100-250mm above board
        
        # Azimuth with J1 bias (more samples on positive side)
        azimuths_negative = np.linspace(-60, 0, 3) * np.pi / 180
        azimuths_positive = np.linspace(0, 90, 6) * np.pi / 180
        azimuths = np.concatenate([azimuths_negative, azimuths_positive])
        
        for dist in distances:
            for height in heights:
                for az in azimuths:
                    if len(poses) >= num_poses:
                        break
                    
                    # Position in X-Y plane around board center
                    x = board_center[0] + dist * np.cos(az)
                    y = board_center[1] + dist * np.sin(az)
                    # Height is absolute above board
                    z = board_center[2] + height  # board is at ~0, so this puts gripper 100-250mm up
                    
                    # Create look-at transform
                    pose_matrix = self.create_look_at_transform(
                        position=[x, y, z],
                        target=board_center
                    )
                    
                    # Check FOV constraint
                    if self.board_in_camera_fov(pose_matrix, board_center):
                        poses.append({
                            'pose_matrix': pose_matrix,
                            'type': 'hemisphere',
                            'params': {'distance': dist, 'height': height, 'azimuth': az}
                        })
        
        return poses
    
    def get_adaptive_candidate_count(self, iteration: int, board_std: Optional[np.ndarray]) -> int:
        """Determine number of candidates based on confidence level"""
        base_count = 100
        
        if board_std is None or np.max(board_std) > 6.0:
            # Low confidence: more candidates
            count = int(base_count * 1.2)
        elif np.max(board_std) > 3.0:
            # Medium confidence: normal
            count = base_count
        else:
            # High confidence: can use fewer
            count = int(base_count * 0.8)
        
        print(f"  Using {count} candidates based on confidence (std: {board_std})")
        return count

    def get_rotating_anchor_points(self, iteration: int) -> List[List[float]]:
        """Get different anchor points for each iteration"""
        n_anchors = 3
        n_total = len(KNOWN_WORKING_POSITIONS)
        
        # Different selection pattern for each iteration
        if iteration == 0:
            indices = [0, n_total//3, 2*n_total//3]
        elif iteration == 1:
            indices = [2, n_total//3 + 2, 2*n_total//3 + 2]
        elif iteration == 2:
            indices = [4, n_total//3 + 4, 2*n_total//3 + 4]
        else:  # iteration 3
            indices = [1, n_total//2, n_total-1]
        
        # Ensure indices are valid
        indices = [i % n_total for i in indices]
        
        anchor_points = [KNOWN_WORKING_POSITIONS[i] for i in indices]
        print(f"  Using anchor points at indices: {indices}")
        return anchor_points

    # PHASE 3: Replace the old update_board_location_weighted with Kalman-like estimation
    def update_board_location_kalman(self, iteration: int, iteration_data: List) -> bool:
        """
        Update board location using Kalman-like recursive estimation.
        Properly handles measurement uncertainty based on detection quality.
        """
        if not iteration_data:
            print("No data collected this iteration")
            return False
        
        # Extract board positions and weights from this iteration's data
        measurements = []
        weights = []
        
        for data in iteration_data:
            # Calculate board position in base frame
            T_gripper2base = np.eye(4)
            T_gripper2base[:3, :3] = data.R_gripper2base
            T_gripper2base[:3, 3] = data.t_gripper2base.flatten()
            
            T_board2cam = np.eye(4)
            T_board2cam[:3, :3] = data.R_target2cam
            T_board2cam[:3, 3] = data.t_target2cam.flatten()
            
            T_board2gripper = self.T_cam2gripper_current @ T_board2cam
            T_board2base = T_gripper2base @ T_board2gripper
            
            board_pos = T_board2base[:3, 3]
            
            # Calculate quality-based weight
            weight = self.calculate_detection_weight(
                data.corners_detected,
                data.board_distance
            )
            
            measurements.append(board_pos)
            weights.append(weight)
        
        measurements = np.array(measurements)
        weights = np.array(weights)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Weighted measurement
        weighted_measurement = np.average(measurements, weights=weights, axis=0)
        
        # Measurement covariance based on weighted variance and detection quality
        measurement_variance = np.average(
            (measurements - weighted_measurement) ** 2,
            weights=weights,
            axis=0
        )
        
        # Scale measurement noise by average detection quality
        avg_corners = np.average([d.corners_detected for d in iteration_data])
        quality_factor = (24 / avg_corners) ** 2  # More corners = less noise
        
        R = self.base_measurement_noise * quality_factor + np.diag(measurement_variance)
        
        # Initialize on first iteration
        if self.board_position_estimate is None:
            self.board_position_estimate = weighted_measurement
            self.board_position_covariance = R
            self.board_location = self.board_position_estimate
            self.board_std = np.sqrt(np.diag(self.board_position_covariance)) * 1000
            
            print(f"\nInitial board estimate:")
            print(f"  Position: X={self.board_location[0]*1000:.1f}mm, "
                  f"Y={self.board_location[1]*1000:.1f}mm, Z={self.board_location[2]*1000:.1f}mm")
            print(f"  Std Dev:  X={self.board_std[0]:.1f}mm, Y={self.board_std[1]:.1f}mm, Z={self.board_std[2]:.1f}mm")
            return True
        
        # Kalman prediction step (board doesn't move)
        P_predicted = self.board_position_covariance + self.process_noise
        
        # Kalman update step
        S = P_predicted + R  # Innovation covariance
        K = P_predicted @ np.linalg.inv(S)  # Kalman gain
        
        # Innovation (measurement residual)
        innovation = weighted_measurement - self.board_position_estimate
        innovation_magnitude = np.linalg.norm(innovation) * 1000  # Convert to mm
        
        # Trust region check - reject large jumps
        if innovation_magnitude > 5.0:  # 5mm threshold
            print(f"WARNING: Measurement rejected - {innovation_magnitude:.1f}mm from estimate")
            print("  Keeping previous estimate to maintain stability")
            # Increase uncertainty slightly since we rejected a measurement
            self.board_position_covariance = P_predicted * 1.1
        else:
            # Accept the measurement
            self.board_position_estimate = self.board_position_estimate + K @ innovation
            self.board_position_covariance = (np.eye(3) - K) @ P_predicted
            
            print(f"  Innovation: {innovation_magnitude:.1f}mm (accepted)")
        
        # Update the class attributes
        old_position = self.board_location.copy() if self.board_location is not None else None
        self.board_location = self.board_position_estimate
        self.board_std = np.sqrt(np.diag(self.board_position_covariance)) * 1000
        
        # Log the update
        print(f"\nKalman-filtered board location (iteration {iteration}):")
        print(f"  Position: X={self.board_location[0]*1000:.1f}mm, "
              f"Y={self.board_location[1]*1000:.1f}mm, Z={self.board_location[2]*1000:.1f}mm")
        print(f"  Std Dev:  X={self.board_std[0]:.1f}mm, Y={self.board_std[1]:.1f}mm, Z={self.board_std[2]:.1f}mm")
        print(f"  Kalman gain diagonal: {np.diag(K)}")
        print(f"  Based on {len(measurements)} measurements with avg {avg_corners:.1f} corners")
        
        if old_position is not None:
            change_mm = np.linalg.norm(self.board_location - old_position) * 1000
            print(f"  Position change: {change_mm:.1f}mm")
        
        # Store history
        self.board_location_history.append({
            'iteration': iteration,
            'position': self.board_location.copy(),
            'std': self.board_std.copy(),
            'n_detections': len(measurements),
            'avg_corners': avg_corners,
            'innovation_mm': innovation_magnitude
        })
        
        return True

    def track_iteration_metrics(self, iteration: int, candidates: List, valid: List, detected_data: List) -> Dict:
        """Track and display iteration performance metrics"""
        metrics = {
            'iteration': iteration,
            'candidates_generated': len(candidates),
            'valid_ik': len(valid),
            'successful_detections': len(detected_data),
            'detection_rate': len(detected_data) / len(valid) if valid else 0,
            'board_std_mm': self.board_std.copy() if self.board_std is not None else None,
            'board_position_mm': self.board_location * 1000 if self.board_location is not None else None
        }
        
        self.iteration_metrics.append(metrics)
        
        # Display metrics
        print(f"\n=== Iteration {iteration} Metrics ===")
        print(f"  Candidates generated: {metrics['candidates_generated']}")
        print(f"  Valid IK solutions: {metrics['valid_ik']}")
        print(f"  Successful detections: {metrics['successful_detections']}")
        print(f"  Detection rate: {metrics['detection_rate']*100:.1f}%")
        
        # Compare to targets
        target_rates = [0.25, 0.45, 0.55, 0.65]  # Mid-points of target ranges
        if iteration < len(target_rates):
            target = target_rates[iteration]
            if metrics['detection_rate'] >= target:
                print(f"  ✓ Exceeding target ({target*100:.0f}%)")
            else:
                print(f"  → Below target ({target*100:.0f}%), but continuing...")
        
        return metrics
    
    def generate_refinement_poses(self, num_poses: int) -> List[Dict]:
        """Generate poses around successful previous poses"""
        poses = []
        
        if not self.successful_poses:
            return poses
        
        # Select best poses to refine around
        n_base = min(5, len(self.successful_poses))
        base_poses = self.successful_poses[-n_base:]  # Use most recent successful
        
        for base_pose in base_poses:
            if len(poses) >= num_poses:
                break
            
            # Small perturbations around successful pose
            for _ in range(num_poses // n_base):
                # Random small changes
                delta_pos = np.random.randn(3) * 0.03  # 30mm std dev
                delta_rot = np.random.randn(3) * 10 * np.pi / 180  # 10 degree std dev
                
                # Apply perturbation
                new_pose = base_pose['pose_matrix'].copy()
                new_pose[:3, 3] += delta_pos
                
                # Apply rotation perturbation
                R_delta = self.euler_to_rotation_matrix(delta_rot)
                new_pose[:3, :3] = new_pose[:3, :3] @ R_delta
                
                poses.append({
                    'pose_matrix': new_pose,
                    'type': 'refinement',
                    'params': {'base_idx': base_poses.index(base_pose)}
                })
        
        return poses
    
    def validate_and_filter_poses(self, candidates: List[Dict], max_poses: int = 20) -> List[Dict]:
        """Validation with PAROL6-appropriate manipulability thresholds"""
        print(f"\nValidating {len(candidates)} candidate poses...")
        
        valid_poses = []
        rejection_reasons = defaultdict(int)
        
        # Get current joint angles
        current_angles = get_robot_joint_angles()
        if not current_angles:
            current_angles = SAFE_START_POSITION
        
        for i, candidate in enumerate(candidates):
            if len(valid_poses) >= max_poses:
                break
            
            pose_matrix = candidate['pose_matrix']
            
            # Validate with IK FIRST
            success, joint_angles, reason = self.validate_pose_with_ik(pose_matrix, current_angles)
            if not success:
                rejection_reasons[f'IK: {reason}'] += 1
                continue
            
            # Calculate manipulability but be more lenient
            manipulability = self.calculate_manipulability(joint_angles)
            
            # PAROL6-specific: Lower threshold for close-to-base operations
            # The robot naturally has low manipulability when working near its base
            SEVERE_SINGULARITY_THRESHOLD = 0.001  # Only reject if VERY close to singularity
            LOW_MANIPULABILITY_WARNING = 0.01     # Warn but accept if above severe threshold
            
            if manipulability < SEVERE_SINGULARITY_THRESHOLD:
                rejection_reasons['Severe singularity'] += 1
                continue

            if joint_angles[2] < 134:
                rejection_reasons['J3 too low, likely to hit camera mount'] += 1
                continue
            
            # Add to valid poses
            candidate['joint_angles'] = joint_angles
            candidate['manipulability'] = manipulability
            valid_poses.append(candidate)
            
            # Log acceptance with appropriate warning
            if manipulability < LOW_MANIPULABILITY_WARNING:
                print(f"  Pose {len(valid_poses)}: Low manipulability ({manipulability:.4f}) but acceptable")
            else:
                print(f"  Pose {len(valid_poses)}: Good manipulability ({manipulability:.4f})")
        
        print(f"\nValidation complete: {len(valid_poses)}/{len(candidates)} poses valid")
        
        if rejection_reasons:
            print("Rejection reasons:")
            for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {reason}: {count}")
        
        # If we got some valid poses but not many, that's still progress!
        if 0 < len(valid_poses) < 5:
            print(f"\n[WARNING] Only {len(valid_poses)} valid poses found (target: 5+)")
            print("  Consider:")
            print("  - Moving calibration board slightly")
            print("  - Adjusting board angle")
            print("  - Using wider pose generation bounds")
        
        return valid_poses
    
    def is_novel_pose(self, pose_matrix: np.ndarray, position_threshold: float = 0.03) -> bool:
        """Check if pose is sufficiently different from history"""
        for hist_pose in self.pose_history:
            # Check position difference
            pos_diff = np.linalg.norm(pose_matrix[:3, 3] - hist_pose['pose_matrix'][:3, 3])
            if pos_diff < position_threshold:
                return False
        return True
    
    def has_rotation_diversity(self, pose_matrix: np.ndarray) -> bool:
        """Check if pose has sufficient rotation diversity from history"""
        min_rotation_rad = MIN_ROTATION_DIVERSITY * np.pi / 180
        
        for hist_pose in self.pose_history:
            R_new = pose_matrix[:3, :3]
            R_hist = hist_pose['pose_matrix'][:3, :3]
            
            # Calculate rotation difference
            R_diff = R_new @ R_hist.T
            angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
            
            if angle_diff < min_rotation_rad:
                return False
        
        return True
    
    def _create_camera_pose(self, camera_pos, board_pos, T_gripper2cam, pose_type='unknown'):
        """Helper to create camera pose looking at board."""
        # Camera looks at board
        z_axis = (board_pos - camera_pos)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Choose arbitrary up vector
        up = np.array([0, 0, 1])
        x_axis = np.cross(up, z_axis)
        if np.linalg.norm(x_axis) < 0.001:
            up = np.array([1, 0, 0])
            x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # Build camera pose matrix
        T_camera = np.eye(4)
        T_camera[:3, 0] = x_axis
        T_camera[:3, 1] = y_axis
        T_camera[:3, 2] = z_axis
        T_camera[:3, 3] = camera_pos
        
        # Transform to gripper pose
        T_gripper = T_camera @ T_gripper2cam
        
        return {
            'pose_matrix': T_gripper,
            'type': pose_type,
            'params': {
                'camera_distance': np.linalg.norm(camera_pos - board_pos)
            }
        }
    
    def validate_pose_with_ik(self, target_pose: np.ndarray, current_angles: List[float]) -> Tuple[bool, Optional[List[float]], str]:
        """Fixed validation with better parsing and error handling"""
        try:
            if self.ik_socket is None:
                self.ik_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.ik_socket.settimeout(0.5)
            
            # Prepare message
            matrix_str = ','.join(map(str, target_pose.flatten()))
            angles_str = ','.join(map(str, current_angles))
            message = f"{matrix_str};{angles_str}"
            
            # Send and receive
            self.ik_socket.sendto(message.encode('utf-8'), ('127.0.0.1', 65432))
            data, _ = self.ik_socket.recvfrom(1024)
            response = data.decode('utf-8').strip()  # Strip whitespace
            
            # CRITICAL FIX: Check for failure FIRST, before trying to parse
            if response == "FAIL" or response == "ERROR" or response.startswith("FAIL"):
                return False, None, "IK not found"
            
            # Try to parse joint angles
            try:
                # Response should be comma-separated joint angles in degrees
                angles = [float(v.strip()) for v in response.split(',')]
                
                # Verify we got 6 joint angles
                if len(angles) != 6:
                    return False, None, f"Invalid response: got {len(angles)} joints, expected 6"
                
                # Check if angles are within limits (in degrees)
                joint_limits = [
                    [-123, 123],      # J1
                    [-145, -3.4],     # J2  
                    [108, 288],       # J3
                    [-105, 105],      # J4
                    [-90, 90],        # J5
                    [0, 360]          # J6
                ]
                
                for i, angle in enumerate(angles):
                    if not (joint_limits[i][0] <= angle <= joint_limits[i][1]):
                        # Still return false but with more detail
                        return False, None, f"Joint {i+1} out of limits: {angle:.1f}°"
                
                # Success! All checks passed
                return True, angles, "Success"
                
            except (ValueError, IndexError) as e:
                # If we can't parse as angles and it's not FAIL/ERROR, something's wrong
                return False, None, f"Parse error: {str(e)} (response: {response[:50]})"
                
        except socket.timeout:
            return False, None, "IK timeout"
        except Exception as e:
            return False, None, f"IK error: {str(e)}"
    
    def calculate_manipulability(self, joint_angles: List[float]) -> float:
        """Calculate manipulability index"""
        try:
            if self.ik_socket is None:
                self.ik_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.ik_socket.settimeout(0.5)
            
            # Send joint angles only (no semicolon)
            message = ','.join(map(str, joint_angles))
            self.ik_socket.sendto(message.encode('utf-8'), ('127.0.0.1', 65432))
            
            data, _ = self.ik_socket.recvfrom(1024)
            return float(data.decode('utf-8'))
        except:
            return 0.5  # Default if service fails
    
    # MODIFIED: Update data collection to track spatial memory
    def collect_calibration_data(self, poses: List[Dict]) -> List:
        """
        Collect calibration data with spatial memory tracking.
        """
        print(f"\nCollecting calibration data from {len(poses)} poses...")
        
        collected = []
        
        cv2.namedWindow("Live Calibration View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Live Calibration View", 800, 600)
        
        for i, pose in enumerate(poses):
            print(f"\n[{i+1}/{len(poses)}] Moving to pose (type: {pose['type']})...")
            
            # Estimate detection probability for this pose
            est_probability = self.get_detection_probability(pose['pose_matrix'])
            print(f"  Estimated detection probability: {est_probability:.1%}")
            
            # Move robot
            result = move_robot_joints(pose['joint_angles'], speed_percentage=20, 
                                    wait_for_ack=True, timeout=15.0)
            
            if not self.wait_for_movement_completion(result, settle_time=1.5):
                print("  [WARNING] Movement incomplete")
                # Still track this as a failed attempt
                self.update_detection_memory(pose['pose_matrix'], 0, False)
                continue
            
            # Get robot pose
            T_gripper2base = get_robot_pose_matrix()
            if T_gripper2base is None:
                print("  [ERROR] Failed to get robot pose")
                self.update_detection_memory(pose['pose_matrix'], 0, False)
                continue
            T_gripper2base = T_gripper2base.astype(np.float64)
            
            # Capture and detect board
            image = self.camera.capture_image()
            T_board2cam, debug_img, corners = self.detect_board_and_estimate_pose(image)
            
            # Update spatial memory
            success = T_board2cam is not None and corners >= 6
            self.update_detection_memory(T_gripper2base, corners if corners else 0, success)
            
            # Show live view with annotations
            display_img = image.copy()
            
            text_color = (0, 255, 0) if success else (0, 0, 255)
            status = f"Pose {i+1}/{len(poses)} | Corners: {corners if corners else 0} | Est.Prob: {est_probability:.1%}"
            cv2.putText(display_img, status, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            
            if debug_img is not None:
                display_img = debug_img
            
            cv2.imshow("Live Calibration View", display_img)
            cv2.waitKey(100)
            
            if success:
                distance = np.linalg.norm(T_board2cam[:3, 3]) * 1000
                
                # Store calibration data
                data = CalibrationData(
                    R_gripper2base=T_gripper2base[:3, :3],
                    t_gripper2base=T_gripper2base[:3, 3].reshape(3, 1),
                    R_target2cam=T_board2cam[:3, :3],
                    t_target2cam=T_board2cam[:3, 3].reshape(3, 1),
                    joint_angles=pose['joint_angles'],
                    board_distance=distance,
                    corners_detected=corners
                )
                
                collected.append(data)
                
                # Track successful pose
                self.successful_poses.append({
                    'pose_matrix': T_gripper2base,
                    'joint_angles': pose['joint_angles'],
                    'corners': corners,
                    'distance': distance
                })
                
                weight = self.calculate_detection_weight(corners, distance)
                print(f"  SUCCESS: {corners} corners at {distance:.1f}mm (weight: {weight:.3f})")
            else:
                print(f"  FAILED: Board not detected or too few corners ({corners if corners else 0})")
        
        cv2.destroyWindow("Live Calibration View")
        print(f"\nData collection complete: {len(collected)}/{len(poses)} successful")
        
        # Print spatial memory statistics
        if self.detection_memory:
            total_cells = len(self.detection_memory)
            successful_cells = sum(1 for cell in self.detection_memory.values() 
                                 if cell['successes'] > 0)
            print(f"Spatial memory: {successful_cells}/{total_cells} cells with successful detections")
        
        return collected
    
    def perform_calibration(self, all_data: List[CalibrationData]) -> Tuple[Optional[np.ndarray], Dict]:
        """Perform hand-eye calibration with multiple methods"""
        print(f"\nPerforming calibration with {len(all_data)} data points...")
        
        # Extract data arrays
        R_gripper2base = [d.R_gripper2base for d in all_data]
        t_gripper2base = [d.t_gripper2base for d in all_data]
        R_target2cam = [d.R_target2cam for d in all_data]
        t_target2cam = [d.t_target2cam for d in all_data]
        
        # Try multiple methods
        methods = [
            (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis"),  # Best accuracy
            (cv2.CALIB_HAND_EYE_PARK, "Park"),
            (cv2.CALIB_HAND_EYE_HORAUD, "Horaud")
        ]
        
        results = []
        
        for method, method_name in methods:
            try:
                R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam,
                    method=method
                )
                
                # Verify rotation matrix
                if self.verify_rotation_matrix(R_cam2gripper):
                    translation = t_cam2gripper.flatten() * 1000
                    
                    results.append({
                        'method': method_name,
                        'R': R_cam2gripper,
                        't': t_cam2gripper,
                        'translation_mm': translation
                    })
                    
                    print(f"  {method_name}: X={translation[0]:.1f}mm, "
                          f"Y={translation[1]:.1f}mm, Z={translation[2]:.1f}mm")
                else:
                    print(f"  {method_name}: Invalid rotation matrix")
                    
            except Exception as e:
                print(f"  {method_name}: Failed - {e}")
        
        if not results:
            return None, {}
        
        # Use Daniilidis if available, otherwise first valid result
        best_result = results[0]
        for r in results:
            if r['method'] == "Daniilidis":
                best_result = r
                break
        
        # Create transformation matrix
        T_cam2gripper = np.eye(4, dtype=np.float64)
        T_cam2gripper[:3, :3] = best_result['R']
        T_cam2gripper[:3, 3] = best_result['t'].flatten()
        
        statistics = {
            'method_used': best_result['method'],
            'num_poses': len(all_data),
            'translation_mm': best_result['translation_mm'],
            'all_methods': results
        }
        
        return T_cam2gripper, statistics
    
    def verify_rotation_matrix(self, R: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Verify rotation matrix is orthonormal"""
        should_be_identity = R @ R.T
        error = np.linalg.norm(np.eye(3) - should_be_identity)
        det = np.linalg.det(R)
        return error < tolerance and abs(det - 1.0) < tolerance
    
    def calculate_calibration_change(self, T_new: np.ndarray, T_old: np.ndarray) -> float:
        """Calculate translation change between calibrations in mm"""
        return np.linalg.norm(T_new[:3, 3] - T_old[:3, 3]) * 1000
    
    def wait_for_movement_completion(self, result: Dict, settle_time: float = 1.0) -> bool:
        """Wait for robot movement to complete using built-in tracking"""
        # If not using tracking, fall back to delay
        if not isinstance(result, dict):
            # Old-style response without tracking, just wait
            delay_robot(settle_time)
            return True
        
        # Check the tracked result
        status = result.get('status', 'UNKNOWN')
        
        if status == 'COMPLETED':
            # Movement completed successfully, add settling time
            if settle_time > 0:
                delay_robot(settle_time)
                time.sleep(settle_time)
            return True
        elif status == 'TIMEOUT':
            print(f"Movement timeout: {result.get('details', 'Unknown error')}")
            return False
        elif status == 'FAILED':
            print(f"Movement failed: {result.get('details', 'Unknown error')}")
            return False
        elif status == 'CANCELLED':
            print(f"Movement cancelled: {result.get('details', 'Unknown error')}")  
            return False
        else:
            # Unknown status, assume success with settling time
            if settle_time > 0:
                delay_robot(settle_time)
            return True
    
    def update_visualization(self, image: np.ndarray, corners: int, status: str):
        """Update live visualization window"""
        display_img = image.copy()
        
        # Add text overlays
        text_lines = [
            f"Status: {status}",
            f"Corners: {corners}/35",
            f"Poses collected: {len(self.all_collected_data)}",
            f"Successful poses: {len(self.successful_poses)}"
        ]
        
        if self.board_std is not None:
            text_lines.append(f"Board std: {self.board_std[0]:.1f}, {self.board_std[1]:.1f}, {self.board_std[2]:.1f}mm")
        
        y_offset = 30
        for line in text_lines:
            cv2.putText(display_img, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        # Color code based on detection quality
        if corners >= 10:
            border_color = (0, 255, 0)  # Green
        elif corners >= 6:
            border_color = (0, 165, 255)  # Orange
        else:
            border_color = (0, 0, 255)  # Red
            
        cv2.rectangle(display_img, (0, 0), 
                     (display_img.shape[1]-1, display_img.shape[0]-1),
                     border_color, 3)
        
        cv2.imshow(self.viz_window, display_img)
        cv2.waitKey(1)
    
    def board_in_camera_fov(self, pose_matrix: np.ndarray, board_center: np.ndarray) -> bool:
        """Check if board would be in camera FOV"""
        # RealSense D435i: ~87° horizontal FOV
        camera_z_axis = pose_matrix[:3, 2]
        to_board = board_center - pose_matrix[:3, 3]
        to_board_normalized = to_board / np.linalg.norm(to_board)
        
        angle = np.arccos(np.clip(np.dot(camera_z_axis, to_board_normalized), -1, 1))
        max_fov_angle = 87 * 0.8 / 2 * np.pi / 180  # Use 80% of FOV for safety
        
        return angle < max_fov_angle
    
    def create_look_at_transform(self, position: List[float], target: np.ndarray) -> np.ndarray:
        """Create transformation matrix that looks at target from position"""
        position = np.array(position)
        
        # Calculate look direction
        z_axis = target - position
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Choose up vector
        up = np.array([0, 0, 1])
        if abs(np.dot(z_axis, up)) > 0.99:
            up = np.array([1, 0, 0])
        
        # Calculate right and up vectors
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        
        # Build transformation matrix
        T = np.eye(4, dtype=np.float64)
        T[:3, 0] = x_axis
        T[:3, 1] = y_axis
        T[:3, 2] = z_axis
        T[:3, 3] = position
        
        return T
    
    def euler_to_rotation_matrix(self, euler_angles: np.ndarray) -> np.ndarray:
        """Convert Euler angles to rotation matrix"""
        rx, ry, rz = euler_angles
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx

    def save_results(self, T_cam2gripper: np.ndarray, statistics: Dict):
        """Save calibration results with organized folder structure"""
        
        # Create timestamp for unique naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create folder structure
        base_dir = "Results"
        calibration_dir = os.path.join(base_dir, "calibration")
        logs_dir = os.path.join(base_dir, "logs")
        #debug_dir = os.path.join(base_dir, "debug_images")  # If needed
        
        # Create directories if they don't exist
        os.makedirs(calibration_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        #os.makedirs(debug_dir, exist_ok=True)
        
        # Generate filenames with timestamp
        npz_filename = f"calibration_refinement_{timestamp}.npz"
        json_filename = f"calibration_refinement_{timestamp}.json"
        log_filename = f"calibration_log_{timestamp}.txt"
        
        # Full paths
        npz_path = os.path.join(calibration_dir, npz_filename)
        json_path = os.path.join(calibration_dir, json_filename)
        log_path = os.path.join(logs_dir, log_filename)
        
        # Save NPZ file
        np.savez(
            npz_path,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            T_cam2gripper=T_cam2gripper,
            statistics=statistics,
            num_poses_used=len(self.all_collected_data),
            board_location=self.board_location,
            board_std=self.board_std,
            timestamp=timestamp
        )
        print(f"\nCalibration saved to {npz_path}")
        
        # Save JSON for human readability
        json_data = {
            "timestamp": timestamp,
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "T_cam2gripper": T_cam2gripper.tolist(),
            "board_location_mm": (self.board_location * 1000).tolist(),
            "board_std_mm": self.board_std.tolist(),
            "statistics": {
                "method_used": statistics.get('method_used', ''),
                "num_poses": statistics.get('num_poses', 0),
                "translation_mm": {
                    "x": float(statistics.get('translation_mm', [0,0,0])[0]),
                    "y": float(statistics.get('translation_mm', [0,0,0])[1]),
                    "z": float(statistics.get('translation_mm', [0,0,0])[2])
                },
                "all_methods": [
                    {
                        "method": m.get('method', ''),
                        "translation_mm": m.get('translation_mm', [0,0,0]).tolist()
                    }
                    for m in statistics.get('all_methods', [])
                ]
            },
            "pose_history": {
                "total_poses_tried": len(self.pose_history),
                "successful_poses": len(self.successful_poses),
                "unique_new_poses": len([p for p in self.pose_history 
                                        if p.get('type') != 'original'])
            }
        }
        
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"Human-readable results saved to {json_path}")
        
        # Save calibration log
        with open(log_path, "w") as f:
            f.write(f"PAROL6 Hand-Eye Calibration Refinement Log\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"Final Calibration Results:\n")
            trans = T_cam2gripper[:3, 3] * 1000
            f.write(f"  Translation: X={trans[0]:.2f}mm, Y={trans[1]:.2f}mm, Z={trans[2]:.2f}mm\n")
            f.write(f"  Method: {statistics.get('method_used', 'Unknown')}\n\n")
            
            f.write(f"Board Localization:\n")
            f.write(f"  Position: {(self.board_location * 1000).tolist()}\n")
            f.write(f"  Std Dev: {self.board_std.tolist()}\n\n")
            
            f.write(f"Data Collection Summary:\n")
            f.write(f"  Total poses attempted: {len(self.pose_history)}\n")
            f.write(f"  Successful detections: {len(self.successful_poses)}\n")
            f.write(f"  Calibration data points: {len(self.all_collected_data)}\n\n")
            
            f.write(f"Convergence History:\n")
            for i, T in enumerate(self.calibration_results):
                t = T[:3, 3] * 1000
                f.write(f"  Iteration {i+1}: X={t[0]:.2f}, Y={t[1]:.2f}, Z={t[2]:.2f}\n")
            
            if len(self.calibration_results) > 1:
                change = self.calculate_calibration_change(
                    self.calibration_results[-1], 
                    self.calibration_results[-2]
                )
                f.write(f"\nFinal convergence: {change:.2f}mm change\n")
        
        print(f"Calibration log saved to {log_path}")
        
        # Also create a "latest" symlink for easy access (Unix-like systems only)
        try:
            latest_npz = os.path.join(calibration_dir, "calibration_latest.npz")
            latest_json = os.path.join(calibration_dir, "calibration_latest.json")
            
            # Remove old symlinks if they exist
            if os.path.islink(latest_npz):
                os.unlink(latest_npz)
            if os.path.islink(latest_json):
                os.unlink(latest_json)
            
            # Create new symlinks
            os.symlink(npz_filename, latest_npz)
            os.symlink(json_filename, latest_json)
            print(f"\nLatest calibration symlinked as 'calibration_latest.npz/json'")
        except (OSError, NotImplementedError):
            # Windows or permission issues - skip symlinks
            pass
        
    def run(self):
        """Run improved calibration refinement with weighted iterations"""
        print("\n" + "="*60)
        print("IMPROVED CALIBRATION REFINEMENT")
        print("="*60)
        
        try:
            # Load existing calibration
            if not self.load_existing_calibration():
                print("ERROR: Could not load calibration")
                return
            
            # Initialize camera
            self.camera = RealSenseCamera()
            self.camera_matrix, self.dist_coeffs = self.camera.get_intrinsics()
            
            # Setup IK validation
            self.ik_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.ik_server_address = ('localhost', 65432)
            self.ik_socket.settimeout(2.0)
            
            # Move to safe start
            print("\nMoving to safe start position...")
            result = move_robot_joints(SAFE_START_POSITION, speed_percentage=20,
                                    wait_for_ack=True, timeout=15.0)
            self.wait_for_movement_completion(result, settle_time=2.0)
            
            # Establish initial board location
            if not self.establish_robust_board_location():
                print("ERROR: Could not establish board location")
                return
            
            # Main refinement loop
            converged = False
            for iteration in range(MAX_ITERATIONS):
                print(f"\n{'='*60}")
                print(f"ITERATION {iteration + 1}/{MAX_ITERATIONS}")
                print(f"{'='*60}")
                
                # Get adaptive candidate count
                num_candidates = self.get_adaptive_candidate_count(iteration, self.board_std)
                
                # Generate diverse poses
                candidates = self.generate_diverse_poses(iteration, num_candidates)
                
                # Validate ALL candidates first (as requested)
                print(f"\nValidating all {len(candidates)} candidates...")
                valid_poses = self.validate_and_filter_poses(candidates, max_poses=1000)  # Don't limit during validation
                
                # Select best 20 poses from valid ones
                if len(valid_poses) > 20:
                    # Sort by some diversity metric if needed, for now just take first 20
                    valid_poses = valid_poses[:20]
                    print(f"Selected best 20 poses from {len(valid_poses)} valid ones")
                
                if len(valid_poses) < 5:
                    print(f"WARNING: Only {len(valid_poses)} valid poses found, need at least 5")
                    # Increase uncertainty for next iteration
                    if self.board_std is not None:
                        self.board_std = self.board_std * 1.5
                    continue
                
                # Add rotating anchor points
                print(f"\nAdding rotating anchor points for iteration {iteration}...")
                anchor_points = self.get_rotating_anchor_points(iteration)
                for i, anchor_pos in enumerate(anchor_points):
                    print(f"  Moving to anchor point {i+1}/3...")
                    result = move_robot_joints(anchor_pos, speed_percentage=20,
                                            wait_for_ack=True, timeout=15.0)
                    self.wait_for_movement_completion(result, settle_time=1.5)
                    
                    T_gripper2base = get_robot_pose_matrix()
                    if T_gripper2base is None:
                        continue
                    T_gripper2base = T_gripper2base.astype(np.float64)
                    
                    image = self.camera.capture_image()
                    T_board2cam, _, corners = self.detect_board_and_estimate_pose(image)
                    
                    self.update_visualization(image, corners if corners else 0, 
                                            f"Anchor {i+1}/3 - Iteration {iteration+1}")
                    
                    if T_board2cam is not None and corners >= 6:
                        data = CalibrationData(
                            R_gripper2base=T_gripper2base[:3, :3],
                            t_gripper2base=T_gripper2base[:3, 3].reshape(3, 1),
                            R_target2cam=T_board2cam[:3, :3],
                            t_target2cam=T_board2cam[:3, 3].reshape(3, 1),
                            joint_angles=anchor_pos,
                            board_distance=np.linalg.norm(T_board2cam[:3, 3]) * 1000,
                            corners_detected=corners
                        )
                        self.all_collected_data.append(data)
                        print(f"    ✓ Anchor point successful (corners: {corners})")
                    else:
                        print(f"    ✗ Anchor point failed (corners: {corners if corners else 0})")
                
                # Collect data from new poses
                iteration_data = self.collect_calibration_data(valid_poses)
                
                # Track metrics
                metrics = self.track_iteration_metrics(iteration, candidates, valid_poses, iteration_data)
                
                # Update board location with weighting
                if not self.update_board_location_kalman(iteration, iteration_data):
                    print("WARNING: Board location update failed, continuing with previous estimate")
                
                # Add to overall data
                self.all_collected_data.extend(iteration_data)
                
                # Perform calibration
                T_cam2gripper_new, stats = self.perform_calibration(self.all_collected_data)
                
                if T_cam2gripper_new is None:
                    print("ERROR: Calibration failed!")
                    continue
                
                # Check convergence
                if len(self.calibration_results) > 0:
                    change = self.calculate_calibration_change(
                        T_cam2gripper_new, self.calibration_results[-1])
                    
                    print(f"\nCalibration change: {change:.2f}mm")
                    old_trans = self.calibration_results[-1][:3, 3] * 1000
                    new_trans = T_cam2gripper_new[:3, 3] * 1000
                    print(f"  Old: X={old_trans[0]:.1f}, Y={old_trans[1]:.1f}, Z={old_trans[2]:.1f}")
                    print(f"  New: X={new_trans[0]:.1f}, Y={new_trans[1]:.1f}, Z={new_trans[2]:.1f}")
                    
                    if change < CONVERGENCE_THRESHOLD:
                        converged = True
                        print(f"[SUCCESS] CONVERGED after {iteration + 1} iterations!")
                
                self.calibration_results.append(T_cam2gripper_new)
                
                # Continue collecting data even after convergence
                if converged and iteration < MAX_ITERATIONS - 1:
                    print("Continuing to collect more data for robustness...")
            
            # Display final summary
            print("\n" + "="*60)
            print("CALIBRATION SUMMARY")
            print("="*60)
            print(f"Total iterations: {len(self.iteration_metrics)}")
            print(f"Total poses collected: {len(self.all_collected_data)}")
            print("\nPer-iteration detection rates:")
            for m in self.iteration_metrics:
                print(f"  Iteration {m['iteration']}: {m['detection_rate']*100:.1f}% "
                    f"({m['successful_detections']}/{m['valid_ik']})")
            
            # Save final results
            if self.calibration_results:
                final_calibration = self.calibration_results[-1]
                final_trans = final_calibration[:3, 3] * 1000
                
                print(f"\n{'='*60}")
                print("SAVING IMPROVED CALIBRATION")
                print(f"{'='*60}")
                
                # Add metrics to statistics
                stats['iteration_metrics'] = self.iteration_metrics
                stats['board_location_history'] = self.board_location_history
                stats['final_detection_rate'] = self.iteration_metrics[-1]['detection_rate'] if self.iteration_metrics else 0
                
                self.save_results(final_calibration, stats)
                
                print(f"\nFinal calibration: X={final_trans[0]:.1f}mm, Y={final_trans[1]:.1f}mm, Z={final_trans[2]:.1f}mm")
                print(f"Used {len(self.all_collected_data)} total poses")
                print(f"Final detection rate: {stats['final_detection_rate']*100:.1f}%")
        
        except KeyboardInterrupt:
            print("\n\nCalibration interrupted by user")
            stop_robot_movement()
        
        except Exception as e:
            print(f"\nError during calibration: {e}")
            import traceback
            traceback.print_exc()
            stop_robot_movement()
        
        finally:
            # Cleanup
            cv2.destroyAllWindows()
            if self.camera:
                self.camera.close()
            if self.ik_socket:
                self.ik_socket.close()
            
            # Return to safe position
            print("\nReturning to safe position...")
            result = move_robot_joints(SAFE_START_POSITION, speed_percentage=20,
                                    wait_for_ack=True, timeout=15.0)
            self.wait_for_movement_completion(result, settle_time=1.0)

def main():
    """Main entry point"""
    refinement = ImprovedCalibrationRefinement()
    refinement.run()


if __name__ == "__main__":
    main()