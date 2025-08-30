#!/usr/bin/env python3
"""
Board-Based Manual Calibration Adjustment Tool for PAROL6 Robot
Combines board coordinate positioning with manual calibration adjustment
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import time
from datetime import datetime
import json
import os

from Headless.robot_api import (
    move_robot_joints,
    move_robot_pose,
    get_robot_pose_matrix,
    get_robot_pose,
    delay_robot,
    is_robot_stopped,
    stop_robot_movement
)

# ChArUco board parameters
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH = 0.025  # 25mm in meters
MARKER_LENGTH = 0.015  # 15mm in meters
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Safe starting position
SAFE_START_POSITION = [84.709, -87.491, 168.742, -5.006, -63.633, 182.002]

# Test positions on the board (in mm from board origin)
AUTO_TEST_POSITIONS = [
    (0, 0, "Origin"),
    (75, 50, "Center"),
    (0, 50, "Left"),
    (150, 50, "Right"),
    (75, 0, "Bottom"),
    (75, 100, "Top"),
]

# Configuration
HOVER_HEIGHT_MM = 25  # Distance above board
MOVEMENT_DURATION = 3.0  # Seconds for movements
ORIENTATION_MODE = "perpendicular"  # Default gripper orientation

class BoardCalibrationAdjuster:
    def __init__(self, calibration_file=None):
        """Initialize with existing calibration file"""

        calibration_file = "Results\calibration\calibration_refinement_20250828_154933.npz"

        # Find most recent calibration if not specified
        if calibration_file is None:
            calibration_dir = "Results/calibration"
            if os.path.exists(calibration_dir):
                files = [f for f in os.listdir(calibration_dir) if f.endswith('.npz')]
                if files:
                    files.sort()
                    calibration_file = os.path.join(calibration_dir, files[-1])
                    print(f"Using most recent calibration: {calibration_file}")
        
        if calibration_file is None:
            calibration_file = "calibration_results.npz"
        
        # Load calibration
        self.load_calibration(calibration_file)
        
        # Initialize camera
        self.init_camera()
        
        # Create ChArUco board
        self.board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),
                                           SQUARE_LENGTH,
                                           MARKER_LENGTH,
                                           ARUCO_DICT)
        self.board.setLegacyPattern(True)
        
        # Manual adjustments (in mm)
        self.manual_offset_x = 0.0
        self.manual_offset_y = 0.0  
        self.manual_offset_z = 0.0
        
        # Current board detection
        self.T_board2cam_current = None
        self.T_board2base_current = None
        self.T_board2base_reference = None  # Reference position (doesn't change with calibration adjustments)
        
        # Test mode
        self.auto_test_mode = False
        self.current_test_index = 0
        
        # Tracking errors
        self.position_errors = []
        
    def load_calibration(self, filename):
        """Load existing calibration"""
        data = np.load(filename)
        self.camera_matrix = data['camera_matrix'].astype(np.float64)
        self.dist_coeffs = data['dist_coeffs'].astype(np.float64)
        self.T_cam2gripper_original = data['T_cam2gripper'].astype(np.float64).copy()
        self.T_cam2gripper = data['T_cam2gripper'].astype(np.float64)
        
        trans = self.T_cam2gripper[:3, 3] * 1000
        print(f"\nLoaded calibration:")
        print(f"  X={trans[0]:.1f}mm, Y={trans[1]:.1f}mm, Z={trans[2]:.1f}mm")
        
    def init_camera(self):
        """Initialize RealSense camera"""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        profile = self.pipeline.start(config)
        
        # Warm up
        for _ in range(10):
            self.pipeline.wait_for_frames()
        
        print("Camera initialized")
        
    def apply_manual_adjustments(self):
        """Apply manual offsets to calibration"""
        # Reset to original
        self.T_cam2gripper = self.T_cam2gripper_original.copy()
        
        # Apply manual adjustments (convert mm to meters)
        self.T_cam2gripper[0, 3] += self.manual_offset_x / 1000.0
        self.T_cam2gripper[1, 3] += self.manual_offset_y / 1000.0
        self.T_cam2gripper[2, 3] += self.manual_offset_z / 1000.0
        
        trans = self.T_cam2gripper[:3, 3] * 1000
        print(f"\nAdjusted calibration:")
        print(f"  X={trans[0]:.1f}mm (offset: {self.manual_offset_x:+.1f}mm)")
        print(f"  Y={trans[1]:.1f}mm (offset: {self.manual_offset_y:+.1f}mm)")
        print(f"  Z={trans[2]:.1f}mm (offset: {self.manual_offset_z:+.1f}mm)")
        
    def detect_board(self, show_test_positions=False):
        """Detect ChArUco board and update current transforms"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            return None, None
        
        # Convert to numpy array
        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect ChArUco corners
        detector = cv2.aruco.CharucoDetector(self.board)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        
        if charuco_corners is not None and len(charuco_corners) >= 6:
            # Get object points
            obj_points, img_points = self.board.matchImagePoints(charuco_corners, charuco_ids)
            
            if obj_points is not None and len(obj_points) >= 6:
                # Estimate pose
                success, rvec, tvec = cv2.solvePnP(
                    obj_points.astype(np.float64),
                    img_points.astype(np.float64),
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    # Store board-to-camera transform
                    R_board2cam, _ = cv2.Rodrigues(rvec)
                    self.T_board2cam_current = np.eye(4, dtype=np.float64)
                    self.T_board2cam_current[:3, :3] = R_board2cam
                    self.T_board2cam_current[:3, 3] = tvec.flatten()
                    
                    # Calculate board in base frame
                    T_gripper2base = get_robot_pose_matrix()
                    if T_gripper2base is not None:
                        T_gripper2base = T_gripper2base.astype(np.float64)
                        T_board2gripper = self.T_cam2gripper @ self.T_board2cam_current
                        self.T_board2base_current = T_gripper2base @ T_board2gripper
                    
                    # Draw visualization
                    cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
                    cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeffs,
                                     rvec, tvec, 0.05)
                    
                    # Draw test positions if requested
                    if show_test_positions:
                        for x_mm, y_mm, name in AUTO_TEST_POSITIONS:
                            # Convert board position to image
                            board_pt = np.array([[x_mm/1000, y_mm/1000, 0]], dtype=np.float64)
                            img_pt, _ = cv2.projectPoints(board_pt, rvec, tvec, 
                                                         self.camera_matrix, self.dist_coeffs)
                            pt = tuple(img_pt[0][0].astype(int))
                            
                            # Draw circle and label
                            cv2.circle(img, pt, 5, (0, 255, 255), -1)  # Yellow dot
                            cv2.circle(img, pt, 7, (0, 0, 0), 2)  # Black outline
                            cv2.putText(img, name, (pt[0]+10, pt[1]-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    # Add detection info
                    cv2.putText(img, f"Detected: {len(charuco_corners)}/24 corners",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    board_dist = np.linalg.norm(tvec) * 1000
                    cv2.putText(img, f"Distance: {board_dist:.0f}mm",
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Show current calibration offsets
                    cv2.putText(img, f"Offsets: X{self.manual_offset_x:+.0f} Y{self.manual_offset_y:+.0f} Z{self.manual_offset_z:+.0f}",
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    return img, True
        
        # No detection
        cv2.putText(img, "Board not detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img, False
        
    def move_to_board_position(self, board_x_mm, board_y_mm, name="Position"):
        """Move gripper to hover above a specific board coordinate"""
        if self.T_board2base_current is None:
            print("Error: Board not detected. Run 'd' first to detect board.")
            return False
        
        print(f"\nMoving to {name}: ({board_x_mm}mm, {board_y_mm}mm)")
        
        # Convert board position to base frame
        board_point = np.array([board_x_mm/1000, board_y_mm/1000, 0, 1])
        base_point = self.T_board2base_current @ board_point
        
        # Add hover height
        target_position = base_point[:3] * 1000  # Convert to mm
        target_position[2] += HOVER_HEIGHT_MM
        
        # Get current orientation or use default
        current_pose = get_robot_pose()
        if current_pose and ORIENTATION_MODE == "maintain":
            orientation = current_pose[3:6]
        else:
            # Perpendicular orientation (pointing down)
            orientation = [180, 0, 180]
        
        # Create full pose
        target_pose = list(target_position) + list(orientation)
        
        print(f"  Target: X={target_position[0]:.1f} Y={target_position[1]:.1f} Z={target_position[2]:.1f}")
        
        # Move robot
        print(f"  Sending movement command...")
        result = move_robot_pose(target_pose, duration=MOVEMENT_DURATION, wait_for_ack=True)
        
        # Wait for movement with timeout
        timeout = MOVEMENT_DURATION + 2.0
        start_time = time.time()
        while not is_robot_stopped() and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if time.time() - start_time >= timeout:
            print("  WARNING: Movement timeout reached!")
            stop_robot_movement()
        
        # Important: Extra settle time to ensure movement is really complete
        time.sleep(1.0)  # Increased settle time
        
        # Verify position by detecting board again
        img, detected = self.detect_board()
        
        if detected:
            # Calculate where we think the board position should be
            expected_board_in_base = self.T_board2base_current[:3, 3] * 1000
            
            # Calculate actual detected position
            actual_point = np.array([board_x_mm/1000, board_y_mm/1000, 0, 1])
            actual_in_base = self.T_board2base_current @ actual_point
            actual_position = actual_in_base[:3] * 1000
            
            # Get actual robot position
            final_pose = get_robot_pose()
            if final_pose:
                robot_pos = np.array(final_pose[:3])
                
                # Calculate error (difference between where robot is and where it should be)
                error = robot_pos - (actual_position + [0, 0, HOVER_HEIGHT_MM])
                
                print(f"\n  Position Error:")
                print(f"    X: {error[0]:+.1f}mm")
                print(f"    Y: {error[1]:+.1f}mm")
                print(f"    Z: {error[2]:+.1f}mm")
                print(f"    Total: {np.linalg.norm(error):.1f}mm")
                
                # Store error for averaging
                self.position_errors.append(error)
                
                # Display image with error info
                cv2.putText(img, f"Error: X{error[0]:+.0f} Y{error[1]:+.0f} Z{error[2]:+.0f}mm",
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.imshow("Board Detection", img)
                cv2.waitKey(1)
                
                return True
        else:
            print("  Warning: Could not detect board at target position")
            cv2.imshow("Board Detection", img)
            cv2.waitKey(1)
            return False
            
    def run_auto_test(self):
        """Run automatic test sequence through multiple board positions"""
        print("\n" + "="*60)
        print("AUTO TEST MODE")
        print("="*60)
        print("Will test calibration at multiple board positions.")
        print("Adjust calibration between positions as needed.")
        print("Press 'q' to stop, 'n' for next position")
        
        # Ensure we have a reference board position
        if self.T_board2base_reference is None:
            if self.T_board2base_current is not None:
                self.T_board2base_reference = self.T_board2base_current.copy()
                print("Using current board position as reference.")
            else:
                print("Error: No board detected. Please detect board first (press 'd')")
                return
        
        self.position_errors = []
        
        for i, (x, y, name) in enumerate(AUTO_TEST_POSITIONS):
            print(f"\n--- Position {i+1}/{len(AUTO_TEST_POSITIONS)}: {name} ---")
            
            # Store current test position for re-testing
            self._current_test_x = x
            self._current_test_y = y
            self._current_test_name = name
            
            # Move to position
            success = self.move_to_board_position(x, y, name)
            
            if success and len(self.position_errors) > 0:
                # Show current average error
                avg_error = np.mean(self.position_errors, axis=0)
                print(f"\n  Average error so far:")
                print(f"    X: {avg_error[0]:+.1f}mm")
                print(f"    Y: {avg_error[1]:+.1f}mm")
                print(f"    Z: {avg_error[2]:+.1f}mm")
                
                # Suggest corrections
                if np.linalg.norm(avg_error) > 2:
                    print(f"\n  Suggested adjustment:")
                    print(f"    X: {-avg_error[0]:+.1f}mm")
                    print(f"    Y: {-avg_error[1]:+.1f}mm")
                    print(f"    Z: {-avg_error[2]:+.1f}mm")
            
            # Wait for user input
            print("\n  Options: [n]ext, [r]epeat, [a]djust, [q]uit")
            
            while True:
                cmd = input("  Choice: ").strip().lower()
                
                if cmd == 'n':
                    break  # Next position
                elif cmd == 'r':
                    # Repeat current position
                    self.move_to_board_position(x, y, name)
                elif cmd == 'a':
                    # Allow adjustment
                    self.interactive_adjustment()
                    # Re-detect board after returning from adjustment
                    print("Re-detecting board...")
                    img, detected = self.detect_board()
                    if detected:
                        print("Board detected, re-testing position...")
                    # Re-test current position
                    self.move_to_board_position(x, y, name)
                elif cmd == 'q':
                    print("Stopping auto test...")
                    return  # Exit auto test
                else:
                    print("  Invalid option. Use: n, r, a, or q")
        
        # Final summary
        if self.position_errors:
            final_avg = np.mean(self.position_errors, axis=0)
            final_std = np.std(self.position_errors, axis=0)
            
            print("\n" + "="*60)
            print("AUTO TEST COMPLETE")
            print("="*60)
            print(f"Tested {len(self.position_errors)} positions")
            print(f"\nAverage Error:")
            print(f"  X: {final_avg[0]:+.1f}mm ± {final_std[0]:.1f}mm")
            print(f"  Y: {final_avg[1]:+.1f}mm ± {final_std[1]:.1f}mm")
            print(f"  Z: {final_avg[2]:+.1f}mm ± {final_std[2]:.1f}mm")
            print(f"  Total: {np.linalg.norm(final_avg):.1f}mm")
            
            if np.linalg.norm(final_avg) > 2:
                print(f"\nRecommended final adjustment:")
                print(f"  X: {-final_avg[0]:+.1f}mm")
                print(f"  Y: {-final_avg[1]:+.1f}mm")
                print(f"  Z: {-final_avg[2]:+.1f}mm")
    
    def interactive_adjustment(self):
        """Quick adjustment interface during auto test"""
        print("\n--- Quick Adjustment ---")
        print("Enter adjustment in mm (e.g., 'x 5' or 'x -5')")
        print("Commands: x/y/z [value], h=home, t=test current, d=done")
        
        # Store current test position for re-testing
        current_test_position = None
        if hasattr(self, '_current_test_x'):
            current_test_position = (self._current_test_x, self._current_test_y, self._current_test_name)
        
        while True:
            cmd = input("Adjust: ").strip().lower()
            
            if cmd == 'd':
                break
            elif cmd == 'h':
                print("Moving to safe position...")
                move_robot_joints(SAFE_START_POSITION, speed_percentage=20, wait_for_ack=True)
                time.sleep(2)
                print("Detecting board...")
                img, detected = self.detect_board(show_test_positions=True)  # Show test positions
                if detected:
                    print("Board detected. Press 't' to return to test position.")
                cv2.imshow("Board Detection", img)
            elif cmd == 't' and current_test_position:
                # Return to current test position
                x, y, name = current_test_position
                print(f"Returning to {name}...")
                self.move_to_board_position(x, y, name)
            elif cmd.startswith('x '):
                try:
                    value = float(cmd.split()[1])
                    self.manual_offset_x += value
                    print(f"X offset now: {self.manual_offset_x:+.1f}mm")
                except (ValueError, IndexError):
                    print("Usage: x [value], e.g., 'x 5' or 'x -3.5'")
            elif cmd.startswith('y '):
                try:
                    value = float(cmd.split()[1])
                    self.manual_offset_y += value
                    print(f"Y offset now: {self.manual_offset_y:+.1f}mm")
                except (ValueError, IndexError):
                    print("Usage: y [value], e.g., 'y 5' or 'y -3.5'")
            elif cmd.startswith('z '):
                try:
                    value = float(cmd.split()[1])
                    self.manual_offset_z += value
                    print(f"Z offset now: {self.manual_offset_z:+.1f}mm")
                except (ValueError, IndexError):
                    print("Usage: z [value], e.g., 'z 5' or 'z -3.5'")
            else:
                print("Commands: x/y/z [value], h=home, t=test current, d=done")
            
            self.apply_manual_adjustments()
            
    def save_calibration(self):
        """Save adjusted calibration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories if needed
        os.makedirs("Results/calibration", exist_ok=True)
        
        # Save NPZ
        npz_file = f"Results/calibration/calibration_manual_{timestamp}.npz"
        np.savez(npz_file,
                T_cam2gripper=self.T_cam2gripper,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs)
        
        # Save JSON with details
        json_file = f"Results/calibration/calibration_manual_{timestamp}.json"
        trans = self.T_cam2gripper[:3, 3] * 1000
        
        data = {
            "timestamp": timestamp,
            "translation_mm": {
                "x": float(trans[0]),
                "y": float(trans[1]),
                "z": float(trans[2])
            },
            "manual_adjustments_mm": {
                "x": float(self.manual_offset_x),
                "y": float(self.manual_offset_y),
                "z": float(self.manual_offset_z)
            },
            "original_translation_mm": {
                "x": float(self.T_cam2gripper_original[0, 3] * 1000),
                "y": float(self.T_cam2gripper_original[1, 3] * 1000),
                "z": float(self.T_cam2gripper_original[2, 3] * 1000)
            }
        }
        
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nCalibration saved:")
        print(f"  NPZ: {npz_file}")
        print(f"  JSON: {json_file}")
        
    def run_interactive(self):
        """Main interactive interface"""
        print("\n" + "="*60)
        print("BOARD-BASED CALIBRATION ADJUSTMENT")
        print("="*60)
        print("\nCommands:")
        print("  d: Detect board")
        print("  a: Run auto test sequence")
        print("  m X Y: Move to board position (e.g., 'm 75 50')")
        print("  x/y/z [value]: Adjust calibration (e.g., 'x 5' or 'y -3.5')")
        print("  r: Reset adjustments")
        print("  s: Save calibration")
        print("  h: Go to safe home position")
        print("  q: Quit")
        print("="*60)
        
        cv2.namedWindow("Board Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Board Detection", 800, 600)
        
        # Move to safe start
        print("\nMoving to safe position...")
        move_robot_joints(SAFE_START_POSITION, speed_percentage=20, wait_for_ack=True)
        time.sleep(2)
        
        while True:
            print(f"\nOffsets: X={self.manual_offset_x:+.1f} Y={self.manual_offset_y:+.1f} Z={self.manual_offset_z:+.1f}")
            cmd = input("Command: ").strip()
            
            if cmd == 'q':
                break
                
            elif cmd == 'd':
                print("Detecting board...")
                img, detected = self.detect_board(show_test_positions=True)  # Show test positions
                if detected:
                    board_pos = self.T_board2base_current[:3, 3] * 1000
                    print(f"Board detected at: X={board_pos[0]:.1f} Y={board_pos[1]:.1f} Z={board_pos[2]:.1f}mm")
                    # Store as reference position for movement calculations
                    self.T_board2base_reference = self.T_board2base_current.copy()
                    print("Board position stored as reference for movements.")
                cv2.imshow("Board Detection", img)
                
            elif cmd == 'a':
                if self.T_board2base_current is None:
                    print("Please detect board first (press 'd')")
                else:
                    self.run_auto_test()
                    
            elif cmd.startswith('m '):
                # Move to specific board coordinate
                parts = cmd.split()
                if len(parts) == 3:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        self.move_to_board_position(x, y, f"Custom ({x},{y})")
                    except ValueError:
                        print("Invalid coordinates. Use: m X Y (e.g., m 75 50)")
                else:
                    print("Usage: m X Y (e.g., m 75 50)")
                    
            elif cmd.startswith('x '):
                try:
                    value = float(cmd.split()[1])
                    self.manual_offset_x += value
                    self.apply_manual_adjustments()
                except (ValueError, IndexError):
                    print("Usage: x [value], e.g., 'x 5' or 'x -3.5'")
                
            elif cmd.startswith('y '):
                try:
                    value = float(cmd.split()[1])
                    self.manual_offset_y += value
                    self.apply_manual_adjustments()
                except (ValueError, IndexError):
                    print("Usage: y [value], e.g., 'y 5' or 'y -3.5'")
                
            elif cmd.startswith('z '):
                try:
                    value = float(cmd.split()[1])
                    self.manual_offset_z += value
                    self.apply_manual_adjustments()
                except (ValueError, IndexError):
                    print("Usage: z [value], e.g., 'z 5' or 'z -3.5'")
                
            elif cmd == 'r':
                self.manual_offset_x = 0.0
                self.manual_offset_y = 0.0
                self.manual_offset_z = 0.0
                self.apply_manual_adjustments()
                print("Adjustments reset")
                
            elif cmd == 's':
                self.save_calibration()
                
            elif cmd == 'h':
                print("Moving to safe position...")
                move_robot_joints(SAFE_START_POSITION, speed_percentage=20, wait_for_ack=True)
                
            else:
                print(f"Unknown command: {cmd}")
        
        cv2.destroyAllWindows()
        self.pipeline.stop()
        
        # IMPORTANT: Stop any pending robot movements
        print("\nStopping robot movements...")
        stop_robot_movement()
        time.sleep(0.5)  # Give time for stop command
        
        print("Calibration adjustment complete!")

if __name__ == "__main__":
    try:
        adjuster = BoardCalibrationAdjuster()
        adjuster.run_interactive()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        stop_robot_movement()  # Emergency stop
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\nError: {e}")
        stop_robot_movement()  # Emergency stop
        cv2.destroyAllWindows()
        raise