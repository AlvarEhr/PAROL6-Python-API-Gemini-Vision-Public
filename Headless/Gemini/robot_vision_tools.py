#!/usr/bin/env python3
"""
Robot Vision Tools for PAROL6
==============================
Vision-based control functions for pick, place, and movement operations.
These functions handle the complex vision processing internally while
presenting simple interfaces to the Gemini Live model.
"""

import asyncio
import time
from typing import List, Dict, Optional, Tuple
import numpy as np

# Import vision and robot control modules
import Headless.Vision.vision_controller as vision
from bounding_box_model import get_boxes_from_pro_model

# Import robot control functions
from Headless.robot_api import (
    move_robot_pose,
    move_robot_joints,
    control_electric_gripper,
    jog_robot_joint,
    get_robot_pose,
    delay_robot,
    get_electric_gripper_status,
    get_robot_joint_angles
)

# Import existing supplementary functions
import supplementary_functions as robot_actions

# ============================================================================
# CONFIGURATION
# ============================================================================

# Pick and place parameters
APPROACH_HEIGHT = 50  # mm above target for approach
GRASP_HEIGHT = 100     # mm lift after grasping
PLACE_OFFSET = 10     # mm above surface when placing
DEFAULT_GRIPPER_ORIENTATION = [90, 180, 0]  # Changed from [0, 180, 0] - rotates gripper 90° around Z

# Add this mapping dictionary after the configuration:
APPROACH_ORIENTATIONS = {
    "vertical": [90, 180, 0],     # Straight down from above
    "angled": [45, 180, 0],       # 45-degree angle approach
    "horizontal": [0, 180, 0],     # Horizontal/side approach
}


# Gripper settings
GRIPPER_OPEN_POS = 0
GRIPPER_CLOSE_POS = 255
GRIPPER_SPEED = 60
GRIPPER_CURRENT = 600  # mA

# Movement speeds (percentage)
SPEED_APPROACH = 40
SPEED_GRASP = 20
SPEED_LIFT = 50  # Increased from 30 for faster lifting
SPEED_TRANSPORT = 50
SPEED_RETURN = 40  # New speed for returning to detection position

GRASP_DEPTH_RATIOS = {
    "vertical": 0.4,    # Grasp 40% down from top
    "angled": 0.3,      # Grasp 30% down (less due to angle)
    "horizontal": 0.5   # Grasp at center
}

# Adaptive clearance based on object size
def get_adaptive_table_clearance(object_height: float) -> float:
    """Get table clearance based on object height"""
    if object_height < 20:
        return 5  # 5mm for very small objects
    elif object_height < 30:
        return 7  # 7mm for small objects like 25mm cubes
    else:
        return 10  # 10mm for larger objects
        
MIN_OBJECT_HEIGHT_FOR_DEEP_GRASP = 12  # Reduced from 15mm
MAX_GRASP_DEPTH_RATIO = 0.5  # Reduced from 0.6 for safety

# ============================================================================
# Core robot control functions
# ============================================================================

class RobotVisionTools:
    """Encapsulates vision-based robot control functions"""
    
    def __init__(self, vision_controller):
        """Initialize with enhanced object tracking"""
        self.vision_controller = vision_controller
        self.object_in_gripper = None
        self.last_pick_position = None
        self.last_place_position = None
        self.is_moving = False
        self.detection_robot_pose = None
        self.detection_robot_angles = None
        
        # NEW: Track picked object properties
        self.picked_object_height = None
        self.picked_object_description = None
        self.picked_object_confidence = None
        self.picked_object_bbox = None
    
    async def search_for_object(self, object_description: str, pattern: str = "sweep", position = "low") -> Dict:
        """
        Start searching movement to find an object
        
        Args:
            object_description: What to search for
            pattern: "sweep" for horizontal, "vertical" for up/down, "spiral" for circular
        """
        self.is_moving = True
        
        try:
            # Start the search movement (non-blocking - returns immediately)
            # The search will continue in the background while Gemini watches
            robot_actions.search(search_pattern=pattern, position=position)
            
            return {
                "status": "searching",
                "target": object_description,
                "pattern": pattern,
                "position": position,
                "message": "Robot is moving in search pattern. Ensure that YOU, the assistant, call the stop_when_found() when you see the object. Not the user."
            }
            
        except Exception as e:
            self.is_moving = False
            return {"status": "error", "message": str(e)}
    
    async def stop_when_found(self) -> Dict:
        """Stop robot movement when object is detected"""
        from Headless.robot_api import stop_robot_movement
        
        self.is_moving = False
        
        try:
            stop_robot_movement()
            
            # Brief pause to let robot stabilize
            await asyncio.sleep(0.2)
            
            # Capture frame for analysis
            if self.vision_controller:
                color_frame, depth_frame = self.vision_controller.get_frames()
                if color_frame is not None:
                    return {
                        "status": "stopped",
                        "frame_captured": True,
                        "frame_shape": color_frame.shape
                    }
            
            return {"status": "stopped", "frame_captured": False}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def pick_up_object(self,
                         object_description: str,
                         approach_angle: str = "vertical",
                         gripper_orientation: List[float] = None,
                         retry_on_failure: bool = True) -> Dict:
        """
        Pick up an object using vision guidance.
        Now tracks object dimensions for place operations.
        """
        if gripper_orientation is None:
            gripper_orientation = APPROACH_ORIENTATIONS.get(approach_angle, DEFAULT_GRIPPER_ORIENTATION)
        
        try:
            # Store robot pose at time of detection
            self.detection_robot_angles = get_robot_joint_angles()
            
            # Get frames
            color_frame, depth_frame = self.vision_controller.get_frames()
            if color_frame is None or depth_frame is None:
                return {"status": "error", "message": "Camera not available"}
            
            # Detect object
            bboxes = await asyncio.to_thread(
                get_boxes_from_pro_model,
                color_frame,
                object_description
            )
            
            if not bboxes:
                return {"status": "error", "message": f"Cannot find {object_description}"}
            
            bbox = bboxes[0]['box']
            
            # Convert to 3D
            detection = self.vision_controller.bbox_to_3d_position(bbox, depth_frame)
            if detection is None:
                return {"status": "error", "message": "Failed to get 3D position"}
            
            # Workspace reachability validation
            if not detection.is_reachable:
                return {
                    "status": "error",
                    "message": f"Object is not reachable (distance: {detection.distance_from_base:.1f}mm)"
                }
            
            # STORE OBJECT PROPERTIES for place operation
            if hasattr(detection, 'depth_stats') and detection.depth_stats:
                self.picked_object_height = detection.depth_stats.get('object_height', 25)  # Default 25mm
                self.picked_object_confidence = detection.depth_stats.get('confidence', 0.5)
                print(f"[PICK] Storing object height: {self.picked_object_height:.1f}mm (conf={self.picked_object_confidence:.2f})")
            else:
                # Fallback if no depth stats
                self.picked_object_height = 25  # Assume standard cube
                self.picked_object_confidence = 0.3
                print(f"[PICK] No depth stats, using default height: {self.picked_object_height}mm")
            
            self.picked_object_description = object_description
            self.picked_object_bbox = bbox
            
            # Execute pick sequence
            success = await self._execute_pick_sequence(
                detection,
                gripper_orientation,
                approach_angle
            )
            
            if success:
                # Store pick information
                self.last_pick_position = detection.position_3d
                
                # Verify pickup
                verify_result = await self._verify_pickup(
                    detection.position_3d,
                    gripper_orientation,
                    object_description
                )
                
                if verify_result["success"]:
                    self.object_in_gripper = object_description
                    return {
                        "status": "success",
                        "object": object_description,
                        "position": detection.position_3d,
                        "approach": approach_angle,
                        "verification": "confirmed",
                        "object_height": self.picked_object_height  # Include in response
                    }
                elif retry_on_failure:
                    # Clear stored properties on failure
                    self.picked_object_height = None
                    self.picked_object_confidence = None
                    # Retry once if verification failed
                    return await self._retry_pick(
                        object_description,
                        gripper_orientation,
                        approach_angle
                    )
                else:
                    return {
                        "status": "warning",
                        "message": "Pickup may have failed",
                        "verification": "failed"
                    }
            else:
                # Clear stored properties on failure
                self.picked_object_height = None
                self.picked_object_confidence = None
                return {"status": "error", "message": "Pick sequence failed"}
                
        except Exception as e:
            # Clear stored properties on error
            self.picked_object_height = None
            self.picked_object_confidence = None
            return {"status": "error", "message": str(e)}
    
    async def _execute_pick_sequence(self,
                                     detection,
                                     orientation: List[float],
                                     approach_angle: str = "vertical") -> bool:
        """
        Execute pick sequence with correct rotation axis and world-frame awareness.
        """
        try:
            # Store original position
            original_z = float(detection.position_3d[2])
            
            # Optimal grasp position computation
            optimal_position, rotation_adj = self.calculate_optimal_grasp_position(
                detection, approach_angle
            )
            
            # Convert to proper list
            if isinstance(optimal_position, np.ndarray):
                optimal_position = optimal_position.tolist()
            optimal_position = [float(p) for p in optimal_position]
            
            # Validate position
            if any(abs(p) > 1000 for p in optimal_position):
                print(f"ERROR: Position out of range: {optimal_position}")
                return False
            
            # Get table height from world-frame stats
            if hasattr(detection, 'depth_stats') and detection.depth_stats:
                table_z = detection.depth_stats.get('table_surface', 0)
            else:
                table_z = 0  # Fallback
            
            # Safety check - don't go too close to table
            if optimal_position[2] < table_z + 3:
                print(f"ERROR: Target position too close to table (Z={optimal_position[2]:.1f}, Table={table_z:.1f})")
                optimal_position[2] = table_z + 5  # Minimum 5mm clearance
            
            # Approach trajectory calculation
            z_descent = original_z - optimal_position[2]
            print(f"Pick sequence: Original Z={original_z:.1f}, Target Z={optimal_position[2]:.1f}, Descent={z_descent:.1f}mm")
            
            # Apply gripper rotation based on approach angle
            adjusted_orientation = list(orientation)
            
            if abs(rotation_adj) > 15:
                if approach_angle == "vertical":
                    # CRITICAL FIX: For vertical approach (Rx=90, Ry=180, Rz=0), 
                    # adjust Rx (index 0) to rotate gripper claws
                    print(f"Applying gripper rotation to Rx: {rotation_adj:.1f}°")
                    adjusted_orientation[0] = float(adjusted_orientation[0] + rotation_adj)
                    
                    # Normalize Rx to reasonable range
                    while adjusted_orientation[0] > 180:
                        adjusted_orientation[0] -= 360
                    while adjusted_orientation[0] < -180:
                        adjusted_orientation[0] += 360
                        
                elif approach_angle == "angled":
                    # For angled approach, might need both adjustments
                    # This depends on your specific robot configuration
                    print(f"Angled approach rotation adjustment not yet implemented")
                    # TODO: Implement based on your robot's specific kinematics
                    
                else:  # horizontal
                    # For horizontal approach, rotation might be on different axis
                    print(f"Horizontal approach rotation adjustment not yet implemented")
            
            # 1. Open gripper
            print("Opening gripper...")
            result = control_electric_gripper(
                action="move",
                position=0,  # Fully open
                speed=150,
                current=500,
                wait_for_ack=False,
                timeout=3.0
            )
            await asyncio.sleep(0.3)  # Reduced from 0.5 - gripper opening
            
            # 2. Move to approach position
            approach_pos = optimal_position.copy()
            approach_pos[2] = float(approach_pos[2] + APPROACH_HEIGHT)
            
            # Ensure approach is well above table
            min_approach_z = table_z + 50
            if approach_pos[2] < min_approach_z:
                approach_pos[2] = min_approach_z
            
            pose = approach_pos + adjusted_orientation
            pose = [float(p) for p in pose]
            
            print(f"Moving to approach: X={pose[0]:.1f}, Y={pose[1]:.1f}, Z={pose[2]:.1f}")
            print(f"  Orientation: RX={pose[3]:.1f}, RY={pose[4]:.1f}, RZ={pose[5]:.1f}")
            
            result = move_robot_pose(
                pose=pose,
                speed_percentage=SPEED_APPROACH,
                wait_for_ack=True,
                timeout=5.0
            )
            
            if isinstance(result, dict) and result.get('status') == 'FAILED':
                print(f"ERROR: Failed to reach approach position: {result.get('details')}")
                return False
            
            await asyncio.sleep(0.5)  # Reduced from 1.0
            
            # 3. Descend to grasp position
            grasp_pose = optimal_position + adjusted_orientation
            grasp_pose = [float(p) for p in grasp_pose]
            
            # Use slower speed if very close to table
            descent_speed = SPEED_GRASP if (optimal_position[2] - table_z) > 20 else SPEED_GRASP // 2
            
            print(f"Descending to grasp: Z={grasp_pose[2]:.1f} (table at {table_z:.1f}) at speed {descent_speed}%")
            
            result = move_robot_pose(
                pose=grasp_pose,
                speed_percentage=descent_speed,
                wait_for_ack=True,
                timeout=5.0
            )
            
            if isinstance(result, dict) and result.get('status') == 'FAILED':
                # Try slightly higher if failed
                print(f"WARNING: Grasp failed, trying 3mm higher")
                grasp_pose[2] += 3
                result = move_robot_pose(
                    pose=grasp_pose,
                    speed_percentage=descent_speed,
                    wait_for_ack=True,
                    timeout=5.0
                )
                if isinstance(result, dict) and result.get('status') == 'FAILED':
                    print(f"ERROR: Cannot reach grasp position")
                    return False
            
            await asyncio.sleep(0.5)  # Reduced from 1.0
            
            # 4. Close gripper
            print("Closing gripper...")
            result = control_electric_gripper(
                action="move",
                position=GRIPPER_CLOSE_POS,
                speed=GRIPPER_SPEED,
                current=GRIPPER_CURRENT,
                wait_for_ack=False,
                timeout=3.0
            )
            await asyncio.sleep(0.7)  # Reduced from 1.0
            
            # 5. Check gripper feedback
            gripper_status = get_electric_gripper_status()
            if gripper_status and len(gripper_status) > 5:
                object_detected = gripper_status[5]
                print(f"Gripper status: Object detected = {object_detected}")
                
                if object_detected == 0 and (optimal_position[2] - table_z) > 10:
                    # Only try recovery if we have room
                    print("No object detected, attempting shallow recovery")
                    
                    control_electric_gripper(
                        action="move",
                        position=20,
                        speed=100,
                        current=500,
                        wait_for_ack=False
                    )
                    await asyncio.sleep(0.5)
                    
                    # Try just 2mm lower (very conservative near table)
                    recovery_pos = optimal_position.copy()
                    recovery_pos[2] = max(recovery_pos[2] - 2, table_z + 3)
                    
                    recovery_pose = recovery_pos + adjusted_orientation
                    recovery_pose = [float(p) for p in recovery_pose]
                    
                    print(f"Recovery at Z={recovery_pose[2]:.1f}")
                    
                    result = move_robot_pose(
                        pose=recovery_pose,
                        speed_percentage=10,
                        wait_for_ack=True,
                        timeout=5.0
                    )
                    
                    if not (isinstance(result, dict) and result.get('status') == 'FAILED'):
                        await asyncio.sleep(0.5)  # Reduced from 1.0
                        
                        control_electric_gripper(
                            action="move",
                            position=GRIPPER_CLOSE_POS,
                            speed=GRIPPER_SPEED - 20,
                            current=GRIPPER_CURRENT + 300,
                            wait_for_ack=False
                        )
                        await asyncio.sleep(0.7)  # Reduced from 1.0
            
            # 6. Lift object
            lift_pos = optimal_position.copy()
            lift_pos[2] = float(optimal_position[2] + GRASP_HEIGHT)
            
            # Ensure lift is well clear of obstacles
            min_lift_z = table_z + 100
            if lift_pos[2] < min_lift_z:
                lift_pos[2] = min_lift_z
            
            lift_pose = lift_pos + adjusted_orientation
            lift_pose = [float(p) for p in lift_pose]
            
            print(f"Lifting to Z={lift_pose[2]:.1f}")
            
            result = move_robot_pose(
                pose=lift_pose,
                speed_percentage=SPEED_LIFT,
                wait_for_ack=True,
                timeout=5.0
            )
            
            await asyncio.sleep(0.3)  # Reduced from 1.0 - just enough for stabilization
            
            print("Pick sequence completed")
            
            # Return to standard position after successful pick
            print("Returning to standard position...")
            robot_actions.move_to_standard_position()
            
            return True
            
        except Exception as e:
            print(f"Pick sequence error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _verify_pickup(self,
                       original_position: List[float],
                       orientation: List[float],
                       object_description: str) -> Dict:
        """
        Verify pickup using gripper sensor feedback and visual confirmation
        """
        try:
            print("Verifying pickup...")
            
            # First check gripper status
            gripper_status = get_electric_gripper_status()
            
            if gripper_status and len(gripper_status) > 5:
                object_detected = gripper_status[5]
                print(f"Gripper verification: Object detected = {object_detected}")
                
                # Object detected values:
                # 0: No object
                # 1: Object detected (closing)
                # 2: Object detected (opening)
                
                if object_detected in [1, 2]:
                    # Gripper confirms object is held
                    print("Pickup verified: Gripper confirms object")
                    return {"success": True, "reason": "gripper_confirmed"}
                else:
                    # No object detected by gripper - try visual verification
                    print("Gripper reports no object, attempting visual verification")
                    
                    # Move to verification position - HIGHER and BACK for better view
                    check_pos = list(original_position)
                    check_pos[2] += APPROACH_HEIGHT * 2.5  # Much higher for better view
                    check_pos[0] -= 50  # Move back for perspective
                    
                    print(f"Moving to verification position: {check_pos}")
                    pose = check_pos + orientation
                    move_robot_pose(pose, speed_percentage=SPEED_APPROACH)
                    await asyncio.sleep(1)
                    
                    # Visual object presence verification
                    color_frame, _ = self.vision_controller.get_frames()
                    if color_frame is None:
                        print("Cannot get camera frame for verification")
                        return {"success": False, "reason": "no_camera_for_verification"}
                    
                    # Try multiple detection prompts for robustness
                    detection_prompts = [
                        object_description,  # Just the object
                        f"{object_description} on table",
                        f"{object_description} on surface"
                    ]
                    
                    object_found = False
                    for prompt in detection_prompts:
                        print(f"Trying detection with: '{prompt}'")
                        try:
                            bboxes = await asyncio.to_thread(
                                get_boxes_from_pro_model,
                                color_frame,
                                prompt
                            )
                            if bboxes and len(bboxes) > 0:
                                object_found = True
                                print(f"Object detected with prompt: '{prompt}'")
                                break
                        except Exception as e:
                            print(f"Detection failed with '{prompt}': {e}")
                            continue
                    
                    if not object_found:
                        # Object not visible - might be in gripper despite sensor
                        print("Object not visible - assuming successful pickup")
                        return {"success": True, "reason": "object_not_visible_assumed_picked"}
                    else:
                        # Object still visible on surface - pickup failed
                        print("Object still visible on surface - pickup failed")
                        return {"success": False, "reason": "object_still_on_surface"}
            else:
                # Gripper status unavailable
                print("Cannot get gripper status for verification")
                # Do visual check as fallback
                
                # Move to verification position
                check_pos = list(original_position)
                check_pos[2] += APPROACH_HEIGHT * 2
                check_pos[0] -= 30
                
                pose = check_pos + orientation
                move_robot_pose(pose, speed_percentage=SPEED_APPROACH)
                await asyncio.sleep(1)
                
                # Try simple visual detection
                color_frame, _ = self.vision_controller.get_frames()
                if color_frame is not None:
                    try:
                        bboxes = await asyncio.to_thread(
                            get_boxes_from_pro_model,
                            color_frame,
                            object_description
                        )
                        if not bboxes:
                            # Object not visible - likely picked up
                            return {"success": True, "reason": "object_disappeared_likely_picked"}
                        else:
                            # Object still there
                            return {"success": False, "reason": "object_still_visible"}
                    except:
                        # Detection error - assume success
                        return {"success": True, "reason": "cannot_verify_assuming_success"}
                
                return {"success": True, "reason": "cannot_verify_no_gripper_status"}
                
        except Exception as e:
            print(f"Verification error: {e}")
            import traceback
            traceback.print_exc()
            # On error, be conservative and assume failure
            return {"success": False, "reason": f"verification_error: {e}"}
    
    async def _retry_pick(self,
                     object_description: str,
                     orientation: List[float],
                     approach_angle: str = "vertical") -> Dict:  # Added approach_angle parameter
        """Retry picking with synchronous gripper commands"""
        try:
            # Failure analysis for retry
            gripper_status = get_electric_gripper_status()

            # Open gripper (SYNCHRONOUS)
            result = control_electric_gripper(
                action="move",
                position=GRIPPER_OPEN_POS,
                speed=100,
                current=500,
                wait_for_ack=False,
                timeout=3.0
            )
            await asyncio.sleep(0.5)
            
            # IMPORTANT: Return to detection position BEFORE re-detection
            if self.detection_robot_angles:
                print("[RETRY] Returning to detection position for better view...")
                move_robot_joints(self.detection_robot_angles, speed_percentage=SPEED_RETURN, wait_for_ack=True)
                await asyncio.sleep(1.5)  # Give time to stabilize and settle
            else:
                print("[RETRY] Warning: No detection position stored")
            
            # Increase current for retry
            adjusted_current = GRIPPER_CURRENT + 300
            
            # Object re-detection attempt from original viewpoint
            color_frame, depth_frame = self.vision_controller.get_frames()
            bboxes = await asyncio.to_thread(
                get_boxes_from_pro_model,
                color_frame,
                object_description
            )
            
            if not bboxes:
                return {"status": "error", "message": "Lost track of object"}
            
            bbox = bboxes[0]['box']
            detection = self.vision_controller.bbox_to_3d_position(bbox, depth_frame)
            
            if detection and detection.is_reachable:
                # Move down again (slightly slower)
                pose = list(detection.position_3d) + orientation
                move_robot_pose(pose, speed_percentage=SPEED_GRASP - 5)
                await asyncio.sleep(1)
                
                # Try gripping with more current (SYNCHRONOUS)
                result = control_electric_gripper(
                    action="move",
                    position=GRIPPER_CLOSE_POS,
                    speed=GRIPPER_SPEED - 20,
                    current=adjusted_current,
                    wait_for_ack=False,
                    timeout=3.0
                )
                await asyncio.sleep(1)
                
                # Gripper status verification
                gripper_status = get_electric_gripper_status()
                if gripper_status and len(gripper_status) > 5:
                    if gripper_status[5] in [1, 2]:  # Object detected
                        # Lift
                        lift_pos = list(detection.position_3d)
                        lift_pos[2] += GRASP_HEIGHT
                        pose = lift_pos + orientation
                        move_robot_pose(pose, speed_percentage=SPEED_LIFT)
                        await asyncio.sleep(1)
                        
                        # Already returned to detection position before retry, no need to do it again
                        
                        self.object_in_gripper = object_description
                        return {
                            "status": "success",
                            "object": object_description,
                            "approach": approach_angle,  # Include approach angle
                            "retry": True,
                            "gripper_confirmed": True
                        }
                
                return {"status": "error", "message": "Retry failed - no object in gripper"}
                
        except Exception as e:
            return {"status": "error", "message": f"Retry error: {e}"}
    
    async def place_object(self,
                      location_description: str,
                      approach_angle: str = "vertical",
                      gripper_orientation: List[float] = None,
                      verify_placement: bool = False) -> Dict:
        """
        Place the currently held object at target location.
        Now properly accounts for object height when calculating place position.
        """
        if gripper_orientation is None:
            gripper_orientation = APPROACH_ORIENTATIONS.get(approach_angle, DEFAULT_GRIPPER_ORIENTATION)
        
        # Object grip status check
        if self.object_in_gripper is None:
            return {"status": "error", "message": "No object in gripper"}
        
        try:
            # Get frames
            color_frame, depth_frame = self.vision_controller.get_frames()
            if color_frame is None or depth_frame is None:
                return {"status": "error", "message": "Camera not available"}
            
            # Detect target location
            bboxes = await asyncio.to_thread(
                get_boxes_from_pro_model,
                color_frame,
                location_description
            )
            
            if not bboxes:
                return {"status": "error", "message": f"Cannot find {location_description}"}
            
            bbox = bboxes[0]['box']
            
            # Convert to 3D
            detection = self.vision_controller.bbox_to_3d_position(bbox, depth_frame)
            if detection is None:
                return {"status": "error", "message": "Failed to get 3D position of target"}
            
            # Target reachability validation
            if not detection.is_reachable:
                return {
                    "status": "error",
                    "message": f"Target is not reachable (distance: {detection.distance_from_base:.1f}mm)"
                }
            
            # FIX: Calculate place position accounting for object height
            target_surface_z = detection.position_3d[2]
            
            # Get object height with fallback
            object_height = self.picked_object_height if self.picked_object_height else 25
            object_confidence = self.picked_object_confidence if self.picked_object_confidence else 0.3
            
            # Dynamic safety margin based on confidence
            if object_confidence > 0.7:
                safety_margin = 15  # High confidence: can place closer
            elif object_confidence > 0.5:
                safety_margin = 20  # Medium confidence
            else:
                safety_margin = 30  # Low confidence: place higher for safety
            
            # Calculate place height
            place_z = target_surface_z + object_height + safety_margin
            
            print(f"[PLACE] Target surface: {target_surface_z:.1f}mm")
            print(f"[PLACE] Object height: {object_height:.1f}mm (conf={object_confidence:.2f})")
            print(f"[PLACE] Safety margin: {safety_margin}mm")
            print(f"[PLACE] Final place Z: {place_z:.1f}mm")
            
            # Build place position
            place_position = [
                float(detection.position_3d[0]),
                float(detection.position_3d[1]),
                float(place_z)
            ]
            
            # Execute place sequence with pre-release
            success = await self._execute_place_sequence(
                place_position,
                gripper_orientation,
                approach_angle,
                object_height
            )
            
            if success:
                # Store place information
                self.last_place_position = place_position
                
                # Clear object tracking
                self.object_in_gripper = None
                self.picked_object_height = None
                self.picked_object_confidence = None
                self.picked_object_description = None
                self.picked_object_bbox = None
                
                # Build result first
                result = {
                    "status": "success",
                    "object": self.picked_object_description,
                    "location": location_description,
                    "approach": approach_angle
                }
                
                # Verify if requested (this moves to verification position)
                if verify_placement:
                    verification = await self._verify_placement(
                        place_position,
                        gripper_orientation,
                        self.picked_object_description
                    )
                    result["verification"] = verification
                
                # NOW return to standard position ONCE, after everything is done
                print("[PLACE] Returning to standard position...")
                robot_actions.move_to_standard_position()
                
                # Small stabilization delay
                await asyncio.sleep(0.3)
                
                return result
            else:
                return {"status": "error", "message": "Place sequence failed"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _execute_place_sequence(self,
                                                  place_position: List[float],
                                                  orientation: List[float],
                                                  approach_angle: str,
                                                  object_height: float) -> bool:
        """
        Execute place sequence with pre-release to prevent collisions.
        Opens gripper slightly above target before final descent.
        """
        try:
            # 1. Move to approach position (above target)
            approach_pos = place_position.copy()
            approach_pos[2] = float(place_position[2] + APPROACH_HEIGHT)
            
            approach_pose = approach_pos + orientation
            approach_pose = [float(p) for p in approach_pose]
            
            print(f"[PLACE] Moving to approach: Z={approach_pose[2]:.1f}")
            
            result = move_robot_pose(
                pose=approach_pose,
                speed_percentage=SPEED_APPROACH,
                wait_for_ack=True,
                timeout=5.0
            )
            
            if not result:
                print("Failed to reach approach position")
                return False
            
            await asyncio.sleep(0.5)  # Reduced from 1.0
            
            # 2. Move to pre-release position (5mm above place position)
            PRE_RELEASE_OFFSET = 5  # mm
            pre_release_pos = place_position.copy()
            pre_release_pos[2] = float(place_position[2] + PRE_RELEASE_OFFSET)
            
            pre_release_pose = pre_release_pos + orientation
            pre_release_pose = [float(p) for p in pre_release_pose]
            
            print(f"[PLACE] Lowering to pre-release: Z={pre_release_pose[2]:.1f}")
            
            result = move_robot_pose(
                pose=pre_release_pose,
                speed_percentage=SPEED_GRASP,
                wait_for_ack=True,
                timeout=5.0
            )
            
            await asyncio.sleep(0.5)
            
            # 3. Open gripper at pre-release height (prevents collision)
            print("[PLACE] Opening gripper at pre-release height...")
            result = control_electric_gripper(
                action="move",
                position=max(0, GRIPPER_OPEN_POS),
                speed=GRIPPER_SPEED,
                current=300,
                wait_for_ack=False,
                timeout=3.0
            )
            await asyncio.sleep(0.3)  # Wait for gripper to fully open
            
            # 4. Optional: Lower to final place position (can skip this if object already released)
            # Only do this for soft placement if needed
            if object_height < 10:  # Only for very thin objects
                place_pose = place_position + orientation
                place_pose = [float(p) for p in place_pose]
                
                print(f"[PLACE] Soft placement to Z={place_pose[2]:.1f}")
                
                result = move_robot_pose(
                    pose=place_pose,
                    speed_percentage=10,  # Very slow for safety
                    wait_for_ack=True,
                    timeout=3.0
                )
                await asyncio.sleep(0.3)
            
            # 5. Place sequence completed - DON'T move to standard position yet
            # This will be handled by the caller after verification (if needed)
            print("[PLACE] Place sequence completed")
            
            await asyncio.sleep(0.3)  # Small delay for stability
            
            return True
            
        except Exception as e:
            print(f"[PLACE] Error in sequence: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _verify_placement(self,
                               place_position: List[float],
                               orientation: List[float],
                               object_description: str) -> str:
        """
        Verify successful placement by visual inspection
        """
        try:
            # Move to verification position (higher and slightly back)
            verify_pos = list(place_position)
            verify_pos[2] += APPROACH_HEIGHT * 2  # Higher for better view
            verify_pos[0] -= 30  # Slightly back
            
            pose = verify_pos + orientation
            move_robot_pose(pose, speed_percentage=SPEED_APPROACH)
            await asyncio.sleep(1)
            
            # Check if object is at target location
            color_frame, _ = self.vision_controller.get_frames()
            if color_frame is None:
                return "cannot_verify"
            
            # Try to detect placed object
            bboxes = await asyncio.to_thread(
                get_boxes_from_pro_model,
                color_frame,
                f"{object_description} at target location"
            )
            
            if bboxes and len(bboxes) > 0:
                return "confirmed"
            else:
                return "uncertain"
                
        except Exception:
            return "verification_error"
        
    def calculate_optimal_grasp_position(self,
                            detection,
                            approach_angle: str = "vertical") -> Tuple[List[float], float]:
        """
        Calculate optimal grasp position using world-frame statistics.
        Enhanced with better safety margins and fallback handling.
        
        Args:
            detection: Detection3D object with world-frame depth_stats
            approach_angle: "vertical", "angled", or "horizontal"
        
        Returns:
            Tuple of (adjusted_position_3d, gripper_rotation_adjustment)
        """
        position_3d = detection.position_3d.copy()
        rotation_adjustment = 0.0
        
        # Maximum useful grasp depth (gripper claw length)
        MAX_GRASP_DEPTH = 35  # mm (leaving 5mm margin from 40mm claws)
        
        # Log current position
        print(f"[GRASP] Object at: X={position_3d[0]:.1f}, Y={position_3d[1]:.1f}, Z={position_3d[2]:.1f}")
        
        # Check if we have world-frame depth statistics
        if not hasattr(detection, 'depth_stats') or detection.depth_stats is None:
            print("[GRASP] No depth stats available, using conservative grasp")
            # FIX 3: More conservative fallback
            position_3d[2] -= 10  # Conservative 10mm descent (was 15mm)
            return position_3d, rotation_adjustment
        
        stats = detection.depth_stats
        object_height = stats['object_height']
        table_surface = stats['table_surface']  # Actual table Z in world frame
        object_top = stats['object_top']  # Actual top of object in world frame
        confidence = stats.get('confidence', 0)
        valid_point_ratio = stats.get('valid_point_ratio', 0)
        
        print(f"[GRASP] Height={object_height:.1f}mm, Table={table_surface:.1f}mm, Top={object_top:.1f}mm")
        print(f"[GRASP] Confidence={confidence:.2f}, Valid points={valid_point_ratio*100:.1f}%")
        
        # Calculate safe grasp depth from table surface (not median)
        object_median = stats['object_median']
        
        # Better handling of uncertain measurements
        if object_height < 10 or confidence < 0.3 or valid_point_ratio < 0.2:
            print(f"[GRASP] Low confidence detection, using cautious approach")
            
            # For uncertain objects, grasp at fixed height from table
            if object_height < 10:
                # Very small/flat object - grasp close to table
                grasp_height = min(8, object_height * 0.4)  # 40% of height or 8mm max
            else:
                # Uncertain measurement - use conservative 30% of detected height
                grasp_height = min(object_height * 0.3, 15)
            
            position_3d[2] = table_surface + grasp_height
            print(f"[GRASP] Low confidence grasp at {grasp_height:.1f}mm above table")
            
        else:  # Normal objects with good measurements
            # Check if object appears rotated based on bounding box aspect ratio
            bbox = detection.bbox_2d if hasattr(detection, 'bbox_2d') else None
            is_rotated = False
            
            if bbox:
                bbox_width = abs(bbox[2] - bbox[0])
                bbox_height = abs(bbox[3] - bbox[1])
                aspect_ratio = bbox_width / max(bbox_height, 1)
                
                # If aspect ratio is not close to 1.0, object is likely rotated
                is_rotated = aspect_ratio < 0.7 or aspect_ratio > 1.3
                
                if is_rotated:
                    print(f"[GRASP] Object appears rotated (aspect ratio: {aspect_ratio:.2f})")
            
            # Calculate ideal grasp height from table based on object height and approach
            if approach_angle == "vertical":
                # For vertical approach, adjust based on rotation
                if is_rotated:
                    ideal_grasp_ratio = 0.20  # 20% for rotated objects (deeper)
                elif object_height < 30:
                    ideal_grasp_ratio = 0.25  # 25% for small objects
                else:
                    ideal_grasp_ratio = 0.30  # 30% for normal objects
            elif approach_angle == "angled":
                # For angled approach, grasp lower
                ideal_grasp_ratio = 0.20 if is_rotated else 0.25
            else:  # horizontal
                # For horizontal, center vertically (50%)
                ideal_grasp_ratio = 0.5
            
            # Calculate grasp height from table
            ideal_grasp_height = object_height * ideal_grasp_ratio
            
            # Apply constraints
            ideal_grasp_height = min(ideal_grasp_height, MAX_GRASP_DEPTH)
            ideal_grasp_height = max(ideal_grasp_height, 5)  # At least 5mm from table
            
            # Calculate target Z position (height above table)
            target_z = table_surface + ideal_grasp_height
            
            # Ensure we don't go above object top
            if target_z > object_top - 5:
                print(f"[GRASP] WARNING: Target too high, adjusting down")
                target_z = object_top - 5  # At least 5mm into object from top
            
            # Ensure minimum clearance from table using adaptive clearance
            adaptive_clearance = get_adaptive_table_clearance(object_height)
            min_safe_z = table_surface + adaptive_clearance
            if target_z < min_safe_z:
                print(f"[GRASP] Adjusting to maintain table clearance ({adaptive_clearance}mm): {target_z:.1f} -> {min_safe_z:.1f}")
                target_z = min_safe_z
            
            position_3d[2] = target_z
            
            print(f"[GRASP] Height from table={ideal_grasp_height:.1f}mm ({ideal_grasp_ratio*100:.0f}%), Target Z={target_z:.1f}mm")
        
        # Handle approach-specific X/Y adjustments
        if approach_angle == "angled":
            # Move back slightly for angled approach
            xy_offset = min(object_height * 0.3, 20)
            position_3d[0] -= xy_offset * 0.7  # Mostly X
            position_3d[1] -= xy_offset * 0.3  # Some Y
            print(f"[GRASP] Angled approach offset: {xy_offset:.1f}mm")
            
        elif approach_angle == "horizontal":
            # Move to edge for horizontal approach
            position_3d[0] -= 20
            print(f"[GRASP] Horizontal approach: moved 20mm in X")
        
        # Get object rotation if available
        if hasattr(detection, 'object_orientation') and detection.object_orientation is not None:
            rotation_adjustment = detection.object_orientation
            
            # Only apply significant rotations (>15 degrees)
            if abs(rotation_adjustment) < 15:
                rotation_adjustment = 0
            else:
                print(f"[GRASP] Object rotation: {rotation_adjustment:.1f}° (will adjust gripper)")
        
        print(f"[GRASP] Final position: X={position_3d[0]:.1f}, Y={position_3d[1]:.1f}, Z={position_3d[2]:.1f}")
        
        return position_3d, rotation_adjustment

    
    async def approach_object(self,
                             object_description: str,
                             distance_mm: float = 100,
                             orientation: List[float] = None) -> Dict:
        """
        Move closer to an object without picking it up
        """
        if orientation is None:
            orientation = DEFAULT_GRIPPER_ORIENTATION
        
        try:
            # Get frames
            color_frame, depth_frame = self.vision_controller.get_frames()
            if color_frame is None or depth_frame is None:
                return {"status": "error", "message": "Camera not available"}
            
            # Detect object
            bboxes = await asyncio.to_thread(
                get_boxes_from_pro_model,
                color_frame,
                object_description
            )
            
            if not bboxes:
                return {"status": "error", "message": f"Cannot find {object_description}"}
            
            bbox = bboxes[0]['box']
            
            # Convert to 3D
            detection = self.vision_controller.bbox_to_3d_position(bbox, depth_frame)
            if not detection:
                return {"status": "error", "message": "Cannot determine 3D position"}
            
            # Calculate approach position
            approach_pos = list(detection.position_3d)
            approach_pos[2] += distance_mm
            
            # Move to approach position
            pose = approach_pos + orientation
            move_robot_pose(pose, speed_percentage=SPEED_APPROACH)
            
            return {
                "status": "approaching",
                "object": object_description,
                "distance": distance_mm,
                "position": approach_pos
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def turn_to_face(self, object_description: str) -> Dict:
        """
        Rotate robot base to center an object in view
        """
        try:
            # Get frames
            color_frame, depth_frame = self.vision_controller.get_frames()
            if color_frame is None:
                return {"status": "error", "message": "Camera not available"}
            
            # Detect object
            bboxes = await asyncio.to_thread(
                get_boxes_from_pro_model,
                color_frame,
                object_description
            )
            
            if not bboxes:
                return {"status": "error", "message": f"Cannot find {object_description}"}
            
            bbox = bboxes[0]['box']
            
            # Calculate rotation needed
            rotation_angle = self.vision_controller.calculate_rotation_to_center(bbox)
            
            if rotation_angle is None or abs(rotation_angle) < 2:
                return {"status": "already_centered", "object": object_description}
            
            # Rotate base
            jog_robot_joint(0, speed_percentage=20, distance_deg=rotation_angle)
            
            return {
                "status": "turning",
                "object": object_description,
                "angle": rotation_angle
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def analyze_scene(self) -> Dict:
        """Get current scene information"""
        try:
            if self.vision_controller:
                color_frame, depth_frame = self.vision_controller.get_frames()
                if color_frame is not None:
                    return {
                        "status": "scene_captured",
                        "frame_shape": color_frame.shape,
                        "has_depth": depth_frame is not None,
                        "object_in_gripper": self.object_in_gripper
                    }
            
            return {"status": "no_camera"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_status(self) -> Dict:
        """Get current robot status"""
        try:
            pose = get_robot_pose()
            
            return {
                "pose": pose,
                "object_in_gripper": self.object_in_gripper,
                "is_moving": self.is_moving
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

# ============================================================================
# EXPORTED FUNCTIONS (for backward compatibility)
# ============================================================================

# Global instance
_robot_tools = None

def initialize_robot_tools(vision_controller):
    """Initialize the robot tools with a vision controller"""
    global _robot_tools
    _robot_tools = RobotVisionTools(vision_controller)
    return _robot_tools

async def search_for_object(object_description: str, pattern: str = "sweep", position: str = "low") -> Dict:
    """Global wrapper for search_for_object"""
    global _robot_tools
    if _robot_tools is None:
        return {"status": "error", "message": "Robot tools not initialized"}
    return await _robot_tools.search_for_object(object_description, pattern, position)

async def stop_when_found() -> Dict:
    """Global wrapper for stop_when_found"""
    global _robot_tools
    if _robot_tools is None:
        return {"status": "error", "message": "Robot tools not initialized"}
    return await _robot_tools.stop_when_found()

async def pick_up_object(object_description: str, approach_angle: str = "vertical") -> Dict:
    """Global wrapper for pick_up_object"""
    global _robot_tools
    if _robot_tools is None:
        return {"status": "error", "message": "Robot tools not initialized"}
    return await _robot_tools.pick_up_object(object_description, approach_angle)

async def place_object(location_description: str, approach_angle: str = "vertical") -> Dict:
    """Global wrapper for place_object"""
    global _robot_tools
    if _robot_tools is None:
        return {"status": "error", "message": "Robot tools not initialized"}
    return await _robot_tools.place_object(location_description, approach_angle)

async def approach_object(object_description: str, distance_mm: float = 100) -> Dict:
    """Global wrapper for approach_object"""
    global _robot_tools
    if _robot_tools is None:
        return {"status": "error", "message": "Robot tools not initialized"}
    return await _robot_tools.approach_object(object_description, distance_mm)

async def turn_to_face(object_description: str) -> Dict:
    """Global wrapper for turn_to_face"""
    global _robot_tools
    if _robot_tools is None:
        return {"status": "error", "message": "Robot tools not initialized"}
    return await _robot_tools.turn_to_face(object_description)

async def analyze_scene() -> Dict:
    """Global wrapper for analyze_scene"""
    global _robot_tools
    if _robot_tools is None:
        return {"status": "error", "message": "Robot tools not initialized"}
    return await _robot_tools.analyze_scene()

# Re-export existing functions from supplementary_functions
wave = robot_actions.wave_command
search = robot_actions.search
