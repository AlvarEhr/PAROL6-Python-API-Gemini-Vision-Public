#!/usr/bin/env python3
"""
Tool Declarations for Gemini Live API
======================================
Defines all available robot control functions for the Gemini model.
These declarations map to functions in robot_vision_tools.py and robot_api.py
"""

# Tool declarations in the format expected by Gemini Live API
robot_tools = [
    # ============================================================================
    # VISION-BASED CONTROL TOOLS (Primary workflow)
    # ============================================================================
    
    {
        "name": "search_for_object",
        "description": "Start a searching movement pattern to locate an object. The robot will move in the specified pattern while looking for the target.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_description": {
                    "type": "string",
                    "description": "Clear description of what to search for (e.g., 'red block', 'blue cup', 'the marker')"
                },
                "pattern": {
                    "type": "string",
                    "enum": ["sweep", "vertical", "spiral"],
                    "description": "Search pattern to use. 'sweep' for horizontal scan, 'vertical' for up/down, 'spiral' for circular",
                    "default": "sweep"
                }
            },
            "required": ["object_description"]
        }
    },
    
    {
        "name": "stop_when_found",
        "description": "Stop all robot movement immediately when the target object is detected in the camera frame. Captures the current frame for processing.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    
    {
        "name": "pick_up_object",
        "description": "Pick up an object that is currently visible in the camera frame. Automatically handles approach, grasping, lifting, and verification. Will retry once if the initial pickup fails.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_description": {
                    "type": "string",
                    "description": "Description of the object to pick up. Should match what's visible in frame."
                },
                "approach_angle": {
                    "type": "string",
                    "enum": ["vertical", "angled", "horizontal"],
                    "description": "How to approach the object. 'vertical' = straight down from above, 'angled' = 45-degree angle, 'horizontal' = from the side (ONLY USE HORIZONTAL IF OBJECT IS TALL OR ELEVATED)",
                    "default": "vertical"
                },
                "retry_on_failure": {
                    "type": "boolean",
                    "description": "Whether to automatically retry if pickup verification fails",
                    "default": False
                }
            },
            "required": ["object_description"]
        }
    },

    {
        "name": "place_object",
        "description": "Place the currently held object at a target location visible in the camera frame. Automatically handles approach, placement, release, and verification.",
        "parameters": {
            "type": "object",
            "properties": {
                "location_description": {
                    "type": "string",
                    "description": "Description of where to place the object (e.g., 'on the blue mat', 'in the box', 'next to the marker')"
                },
                "approach_angle": {
                    "type": "string",
                    "enum": ["vertical", "angled", "horizontal"],
                    "description": "How to approach the placement location. 'vertical' = straight down, 'angled' = 45-degree angle, 'horizontal' = from the side",
                    "default": "vertical"
                },
                "verify_placement": {
                    "type": "boolean",
                    "description": "Whether to verify successful placement by visual inspection",
                    "default": False
                }
            },
            "required": ["location_description"]
        }
    },

    
    # ============================================================================
    # GENERAL MOVEMENT TOOLS
    # ============================================================================
    
    {
        "name": "approach_object",
        "description": "Move the robot closer to an object without picking it up. Useful for inspection or positioning.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_description": {
                    "type": "string",
                    "description": "Description of the object to approach"
                },
                "distance_mm": {
                    "type": "number",
                    "description": "How close to get to the object in millimeters",
                    "default": 100,
                    "minimum": 50,
                    "maximum": 300
                }
            },
            "required": ["object_description"]
        }
    },
    
    {
        "name": "turn_to_face",
        "description": "Rotate the robot base to center an object in the camera view. Useful for better viewing angle or reachability.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_description": {
                    "type": "string",
                    "description": "Description of the object to face"
                }
            },
            "required": ["object_description"]
        }
    },
    
    # ============================================================================
    # STATUS AND UTILITY TOOLS
    # ============================================================================
    
    {
        "name": "analyze_scene",
        "description": "Analyze the current camera view and report what's visible. Returns information about the scene.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    
    {
        "name": "get_robot_status",
        "description": "Get the current status of the robot including pose, joint angles, and what's in the gripper.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    
    {
        "name": "stop_robot",
        "description": "Emergency stop - immediately halt all robot movement.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    
    # ============================================================================
    # GESTURE/EXPRESSION TOOLS
    # ============================================================================
    
    {
        "name": "wave",
        "description": "Perform a friendly waving gesture. The robot will wave its end effector in a greeting motion.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    
    # ============================================================================
    # DIRECT ROBOT CONTROL (Advanced - only if needed)
    # ============================================================================
    
    {
        "name": "move_robot_joints",
        "description": "Move robot joints to specific angles. Advanced function for direct joint control.",
        "parameters": {
            "type": "object",
            "properties": {
                "joint_angles": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 6,
                    "maxItems": 6,
                    "description": "Target angles for each joint [j1, j2, j3, j4, j5, j6] in degrees"
                },
                "speed_percentage": {
                    "type": "number",
                    "description": "Movement speed as percentage (1-100)",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 30
                }
            },
            "required": ["joint_angles"]
        }
    },
    
    {
        "name": "move_robot_pose",
        "description": "Move robot to a specific pose in Cartesian coordinates. Advanced function for direct pose control.",
        "parameters": {
            "type": "object",
            "properties": {
                "pose": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 6,
                    "maxItems": 6,
                    "description": "Target pose as [x, y, z, Rx, Ry, Rz] where position is in mm and rotation in degrees"
                },
                "speed_percentage": {
                    "type": "number",
                    "description": "Movement speed as percentage (1-100)",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 30
                }
            },
            "required": ["pose"]
        }
    },
    
    {
        "name": "control_gripper",
        "description": "Control the electric gripper directly. Advanced function for manual gripper control.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["open", "close", "move"],
                    "description": "Gripper action: 'open' fully opens, 'close' fully closes, 'move' goes to specific position"
                },
                "position": {
                    "type": "integer",
                    "description": "Target position (0-255) when action is 'move'",
                    "minimum": 0,
                    "maximum": 255
                },
                "speed": {
                    "type": "integer",
                    "description": "Movement speed (0-255)",
                    "minimum": 0,
                    "maximum": 255,
                    "default": 100
                },
                "current": {
                    "type": "integer",
                    "description": "Current limit in mA (100-1000)",
                    "minimum": 100,
                    "maximum": 1000,
                    "default": 500
                }
            },
            "required": ["action"]
        }
    },
    
    {
        "name": "home_robot",
        "description": "Move the robot to its home position. This performs the homing sequence to establish reference positions.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]

# ============================================================================
# TOOL GROUPS (for reference)
# ============================================================================

# Group tools by category for easier reference
TOOL_GROUPS = {
    "vision_control": [
        "search_for_object",
        "stop_when_found", 
        "pick_up_object",
        "place_object",
        "approach_object",
        "turn_to_face"
    ],
    "status": [
        "analyze_scene",
        "get_robot_status"
    ],
    "gestures": [
        "wave"
    ],
    "direct_control": [
        "move_robot_joints",
        "move_robot_pose",
        "control_gripper",
        "home_robot",
        "stop_robot"
    ]
}

# ============================================================================
# SIMPLIFIED TOOL SET (for basic operations)
# ============================================================================

# Minimal tool set for simple pick-and-place operations
SIMPLE_TOOLS = [
    tool for tool in robot_tools 
    if tool["name"] in [
        "search_for_object",
        "stop_when_found",
        "pick_up_object",
        "place_object",
        "wave",
        "stop_robot",
        "get_robot_status"
    ]
]

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_tool_declaration(tool: dict) -> bool:
    """Validate that a tool declaration has required fields"""
    required_fields = ["name", "description", "parameters"]
    for field in required_fields:
        if field not in tool:
            print(f"Missing required field '{field}' in tool: {tool.get('name', 'unknown')}")
            return False
    
    # Validate parameters structure
    params = tool["parameters"]
    if not isinstance(params, dict) or "type" not in params:
        print(f"Invalid parameters structure in tool: {tool['name']}")
        return False
    
    return True

def validate_all_tools() -> bool:
    """Validate all tool declarations"""
    print(f"Validating {len(robot_tools)} tool declarations...")
    
    all_valid = True
    for tool in robot_tools:
        if not validate_tool_declaration(tool):
            all_valid = False
    
    if all_valid:
        print("✓ All tool declarations are valid")
    else:
        print("✗ Some tool declarations have errors")
    
    return all_valid

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def get_tools_for_mode(mode: str = "full") -> list:
    """
    Get appropriate tool set based on mode
    
    Args:
        mode: "full" for all tools, "simple" for basic operations
        
    Returns:
        List of tool declarations
    """
    if mode == "simple":
        return SIMPLE_TOOLS
    else:
        return robot_tools

def get_tool_by_name(name: str) -> dict:
    """Get a specific tool declaration by name"""
    for tool in robot_tools:
        if tool["name"] == name:
            return tool
    return None

def get_tools_by_group(group: str) -> list:
    """Get all tools in a specific group"""
    if group in TOOL_GROUPS:
        tool_names = TOOL_GROUPS[group]
        return [tool for tool in robot_tools if tool["name"] in tool_names]
    return []

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Validate all declarations when run directly
    validate_all_tools()
    
    # Print summary
    print(f"\nTool Summary:")
    print(f"Total tools: {len(robot_tools)}")
    for group, tools in TOOL_GROUPS.items():
        print(f"  {group}: {len(tools)} tools")
    
    # Example: Get simple tools
    simple = get_tools_for_mode("simple")
    print(f"\nSimple mode tools: {len(simple)}")
    for tool in simple:
        print(f"  - {tool['name']}")