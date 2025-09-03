#!/usr/bin/env python3
"""
Gemini Live Controller for PAROL6 Robot
========================================
Supports two operation modes:
- TEXT: Step-by-step text control for testing individual functions
- AUTONOMOUS: Compositional function calling for multi-step operations
"""

import os
import asyncio
import json
import time
import traceback
import warnings
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
import argparse
import pyaudio
import keyboard
import threading
import queue

import cv2
import numpy as np
import PIL.Image
from dotenv import load_dotenv

from google import genai
from google.genai import types

# Import tool declarations
from tool_declarations import robot_tools
import Headless.Vision.vision_controller as vision
import robot_vision_tools

# Import basic robot functions for status and direct control
from Headless.robot_api import (
    stop_robot_movement,
    get_robot_pose,
    get_robot_joint_angles,
    move_robot_joints,
    move_robot_pose,
    control_electric_gripper,
    home_robot,
    is_robot_stopped
)

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 16000
AUDIO_CHUNK = 1024

class OperationMode(Enum):
    """Operating modes"""
    TEXT = "text"  # Step-by-step text control for testing
    AUTONOMOUS = "autonomous"  # Compositional function calling

@dataclass
class GeminiConfig:
    """Configuration for Gemini Live API"""
    mode: OperationMode
    model: str = "gemini-2.5-flash-live-preview"  # Default for text responses
    enable_compositional: bool = False
    turn_coverage: str = "TURN_INCLUDES_ONLY_ACTIVITY"
    snapshot_interval: float = 1.0
    livestream_fps: int = 10
    enable_voice: bool = False  # Whether to use voice RESPONSES (TTS)
    enable_debug: bool = True
    input_mode: str = "text"  # "text" or "audio" INPUT
    push_to_talk_key: str = "space"  # Key to hold for voice input
    response_mode: str = "text"  # "text" or "audio" RESPONSE
    
    @classmethod
    def from_mode(cls, mode_name: str, input_mode: str = "text", response_mode: str = "text"):
        """Create config based on mode
        
        Args:
            mode_name: "text" or "autonomous"
            input_mode: "text" or "audio" for input
            response_mode: "text" or "audio" for responses
        """
        mode = OperationMode(mode_name.lower())
        
        # Determine if we should enable voice based on response mode
        enable_voice_output = (response_mode == "audio")
        
        configs = {
            OperationMode.TEXT: cls(
                mode=mode,
                enable_compositional=False,
                turn_coverage="TURN_INCLUDES_ONLY_ACTIVITY",
                snapshot_interval=1.0,
                livestream_fps=3,
                enable_voice=enable_voice_output,
                enable_debug=True,
                input_mode=input_mode,
                response_mode=response_mode
            ),
            OperationMode.AUTONOMOUS: cls(
                mode=mode,
                enable_compositional=True,
                turn_coverage="TURN_INCLUDES_ALL_INPUT",
                snapshot_interval=0.5,
                livestream_fps=5,
                enable_voice=enable_voice_output,
                enable_debug=True,  # Keep debug for now
                input_mode=input_mode,
                response_mode=response_mode
            )
        }
        
        return configs.get(mode, configs[OperationMode.TEXT])

# ============================================================================
# GEMINI LIVE SESSION MANAGER
# ============================================================================

class GeminiLiveSession:
    """Manages Gemini Live sessions with mode-specific behavior"""
    
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=os.environ.get("GEMINI_API_KEY")
        )
        self.session = None
        self.vision_controller = None
        self.robot_tools = None
        self.latest_frame = None
        self._last_snapshot_time = 0
        self._is_livestreaming = False
        self._movement_active = False
        self._session_active = False
        self._last_position_snapshot_sent = False
        self._last_robot_pose = None
        self._waiting_for_input = False
        self._robot_connected = True
        self._last_robot_check_time = 0
        self._robot_check_interval = 5.0
        self._robot_disconnected_logged = False
        self._robot_checked = False
        self._robot_connected = False
        self._waiting_for_response = False

        # Add audio-related attributes
        self.audio_queue = queue.Queue()
        self.audio_output_queue = None  # Will be asyncio.Queue() when session starts
        self.pya = None
        self.audio_stream = None
        self.audio_output_stream = None  # For audio playback
        self._recording = False
        self._audio_task = None
        self._audio_playback_task = None
        self._audio_frame_for_context = None  # Store frame to send with audio
        
        # For autonomous mode
        self._autonomous_active = False
        self._current_task = None
        self._current_search_target = None
        self._search_message_sent = False

        # ADD these for reconnection support
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 2  # Initial delay in seconds
        self._last_successful_response = time.time()
        self._connection_timeout = 30  # Consider connection dead after 30s of no responses
        self._pending_tool_response = None  # Store pending tool response if connection drops
        
        # Build tool registry
        self.tool_registry = None
    
    def _build_tool_registry(self) -> Dict[str, Any]:
        """Build registry of available tool functions - MUST be called after initialize_components()"""
        if not self.robot_tools:
            raise RuntimeError("Cannot build tool registry before initializing robot tools")
        
        return {
            # Core workflow tools - use the instance methods directly
            "search_for_object": self.search_for_object_wrapper,
            "stop_when_found": self.stop_when_found_wrapper,
            "pick_up_object": self.robot_tools.pick_up_object,
            "place_object": self.robot_tools.place_object,
            
            # General movement tools
            "approach_object": self.robot_tools.approach_object,
            "turn_to_face": self.robot_tools.turn_to_face,
            
            # Utility tools
            "analyze_scene": self.robot_tools.analyze_scene,
            "get_robot_status": self.get_robot_status,
            
            # Fun commands - check if these exist as methods
            "wave": getattr(self.robot_tools, 'wave', robot_vision_tools.wave),
            
            # Direct control
            "stop_robot": stop_robot_movement,
            "move_robot_joints": move_robot_joints,
            "move_robot_pose": move_robot_pose,
            "control_gripper": self._create_gripper_wrapper(),
            "home_robot": home_robot,
        }
    
    def _async_wrapper(self, async_func):
        """Wrapper to handle async functions in the tool registry"""
        return async_func
    
    def _create_sync_wrapper(self, func):
        """Wrapper for synchronous functions"""
        return func
    
    def _create_gripper_wrapper(self):
        """Create a wrapper for gripper control with simplified interface"""
        def gripper_control(action="open", **kwargs):
            if action == "open":
                return control_electric_gripper(action="move", position=100, speed=100, current=500)
            elif action == "close":
                return control_electric_gripper(action="move", position=200, speed=60, current=600)
            else:
                return control_electric_gripper(action=action, **kwargs)
        return gripper_control
    
    async def search_for_object_wrapper(self, object_description: str, pattern: str = "sweep", position: str = "low"):
        """Wrapper that tracks search target"""
        self._current_search_target = object_description
        self._search_message_sent = False
        if self.config.enable_debug:
            print(f"[DEBUG] Starting search for: {object_description}")
        
        # Execute the search (starts movement and returns immediately)
        result = await self.robot_tools.search_for_object(object_description, pattern, position)
        
        # The search is now running in the background
        # Movement will continue until stop_when_found() is called or search completes
        
        return result
        
    async def stop_when_found_wrapper(self):
        """Wrapper that clears search target"""
        result = await self.robot_tools.stop_when_found()
        self._current_search_target = None
        self._search_message_sent = False
        if self.config.enable_debug:
            print("[DEBUG] Search stopped, target cleared")
        return result
    
    def _build_system_instruction(self) -> str:
        """Build system instruction based on operation mode"""
        if self.config.mode == OperationMode.AUTONOMOUS:
            return """You are a helpful assistant who is also an autonomous robot controller with advanced vision capabilities.

AUTONOMOUS OPERATION PROTOCOL:
You have complete authority to execute multi-step operations without waiting for user confirmation.
When given a task, immediately decompose it into steps and execute them sequentially. If the user asks you to pick up or place an object (or both)
and you cannot currently see it, start the search process IMMEDIATELY instead of requesting the user for anything.

To emphasize, if the user asks you to pick up an object and you cannot see it, DO NOT ASK THE USER FOR CLARIFICATION, simply use the search_for_object function to find it.

COMPOSITIONAL FUNCTION CALLING:
- Chain multiple functions to complete complex tasks without waiting for user input. YOU MUST CALL THESE FUNCTIONS YOURSELF, NOT THE USER.
- Example sequence for "pick up the red ball":
1. search_for_object - Start searching
2. stop_when_found - Stop when visible
3. pick_up_object - Pick it up (auto-verifies)
4. search_for_object - Search for placement area
5. stop_when_found - Stop when area visible
6. place_object - Place it

VISUAL PERCEPTION DURING MOVEMENT:
- DO NOT EXPECT USER INPUT WHILE SEARCHING, YOU MUST HANDLE STOPPING ONCE OBJECT IS DETECTED YOURSELF.
- When robot is moving, you receive continuous video frames at 10 FPS
- Analyze EVERY frame during search operations for the target object
- When YOU, the assistant, detect the object AND it is FULLY in frame, IMMEDIATELY call stop_when_found(). REMEMBER TO DO THIS. Make sure that you can actually see the object.
- If you believe you see the object and think it is in frame, STOP IMMEDIATELY. Take some time to confirm you see the object before proceeding.
Otherwise, you can resume searching by calling search_for_object() again.
- If you accidentally or prematurely call stop_when_found(), you can resume searching by calling search_for_object() again.

OBJECT DETECTION PROTOCOL:
- During search_for_object: robot moves in search pattern while you analyze video
- Ensure you can clearly see the object as described by the user before stopping AND that it is FULLY in frame
- Immediately stop when target detected, don't wait for search to complete
- Call to pick_up_object or place_object with a clear description of the object/location

SAFETY RULES:
- Always stop movement if something seems to be wrong
- Stop immediately if any error detected

IF YOU ARE ASKED TO PICK SOMETHING UP OR PLACE IT AND YOU CANNOT SEE IT, DO NOT ASK FOR CLARIFICATION, SIMPLY USE THE search_for_object FUNCTION TO FIND IT.

IF YOU ARE ASKED TO PICK UP AND PLACE SOMETHING SOMETHING BUT HAVE NOT YET PICKED UP AN OBJECT, FIRST PICK IT UP THEN FIND THE AREA TO PLACE IT.

AGAIN, IF YOU HAVE NOT YET PICKED UP AND OBJECT AND CANNOT SEE ITS DESTINATION, FIRST SEARCH FOR THE OBJECT TO PICK UP, 
PICK IT UP AND ONLY SHOULD YOU BEGIN THE SEARCH FOR THE PLACEMENT AREA, ALL WITHOUT ASKING THE USE, 
REMEMBER TO USE THE search_for_object FUNCTION FOR THE PLACEMENT AREA TOO.

Execute complete sequences autonomously. Report only essential info."""
        
        else:  # TEXT mode
            return """You are a helpful assistant that has control over the PAROL6 robot arm with a camera.

You should attempt anything the user asks unless it is unsafe or impossible. You have access to these tools:

PICK AND PLACE WORKFLOW:
1. search_for_object - Start searching
2. stop_when_found - Stop when visible
3. pick_up_object - Pick it up (auto-verifies)
4. search_for_object - Search for placement area
5. stop_when_found - Stop when area visible
6. place_object - Place it

GENERAL MOVEMENT:
- approach_object: Move closer
- turn_to_face: Rotate to center
- wave: Friendly gesture
- analyze_scene: Get scene information

CRITICAL RULES:
- NEVER use code execution or ExecutableCode
- ONLY use the provided tool functions
- For any action, use the appropriate tool function
- If you cannot find an appropriate tool, say so

Report results clearly and wait for next instruction."""
    
    def initialize_components(self):
        """Initialize camera and robot tools"""
        try:
            # Initialize vision controller
            self.vision_controller = vision.VisionController()
            if not self.vision_controller.initialize_camera():
                raise Exception("Failed to initialize camera")
            
            # Start camera preview window
            self.vision_controller.start_display_thread("Gemini Live - Camera View")
            
            # Initialize robot vision tools
            self.robot_tools = robot_vision_tools.initialize_robot_tools(self.vision_controller)
            
            # NOW build the tool registry after robot_tools is initialized
            self.tool_registry = self._build_tool_registry()
            
            print("Components initialized")
            return True
            
        except Exception as e:
            print(f"Failed to initialize components: {e}")
            if self.config.enable_debug:
                traceback.print_exc()
            return False
    
    async def run_session(self):
        """Enhanced run_session with automatic reconnection"""
        while self._reconnect_attempts < self._max_reconnect_attempts:
            try:
                await self._create_and_run_session()
                # If we exit normally, break the loop
                break
                
            except Exception as e:
                error_msg = str(e)
                
                # Connection error retry logic
                if any(err in error_msg.lower() for err in ['1011', 'internal error', 'connection', 'closed']):
                    self._reconnect_attempts += 1
                    
                    if self._reconnect_attempts < self._max_reconnect_attempts:
                        wait_time = self._reconnect_delay * (2 ** (self._reconnect_attempts - 1))
                        print(f"\n[RECONNECT] Connection lost. Attempting reconnection {self._reconnect_attempts}/{self._max_reconnect_attempts} in {wait_time}s...")
                        
                        # Clean up current session
                        await self._cleanup_session()
                        
                        # Wait before reconnecting
                        await asyncio.sleep(wait_time)
                        
                        # Reinitialize components if needed
                        if not self.vision_controller or not self.vision_controller.pipeline:
                            print("[RECONNECT] Reinitializing components...")
                            self.initialize_components()
                        
                        continue
                    else:
                        print(f"[ERROR] Max reconnection attempts reached. Please restart the application.")
                        break
                else:
                    # Non-recoverable error
                    print(f"[ERROR] Non-recoverable error: {error_msg}")
                    if self.config.enable_debug:
                        traceback.print_exc()
                    break
        
        print("[INFO] Session ended")
    
    async def _create_and_run_session(self):
        """Create and run a single session (extracted for reconnection)"""
        # Available tools initialization
        tools = [{"function_declarations": robot_tools}]
        
        # NOTE: Gemini Live API only supports one response modality at a time
        # In audio mode, responses will be audio-only (no text feedback in terminal)
        # Determine the response modality based on configuration
        if self.config.enable_voice:
            response_modality = [types.Modality.AUDIO]  # Audio-only mode
        else:
            response_modality = [types.Modality.TEXT]   # Text-only mode
        
        # Build session config
        config = types.LiveConnectConfig(
            response_modalities=response_modality,  # Use the determined modality
            system_instruction=self._build_system_instruction(),
            tools=tools,
            media_resolution="MEDIA_RESOLUTION_MEDIUM"
        )
        
        # Add mode-specific configurations
        if self.config.mode == OperationMode.AUTONOMOUS:
            config.realtime_input_config = types.RealtimeInputConfig(
                turn_coverage=self.config.turn_coverage
            )
        elif self.config.input_mode == "audio":
            # For audio mode, use TURN_INCLUDES_ALL_INPUT like Google's example
            # This ensures all audio input is considered part of the turn
            config.realtime_input_config = types.RealtimeInputConfig(
                turn_coverage="TURN_INCLUDES_ALL_INPUT"
            )
        
        # Add speech config if using audio modality
        if self.config.enable_voice:
            config.speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Charon"
                    )
                )
            )
        
        print(f"[CONNECT] Establishing connection...")
        print(f"Mode: {self.config.mode.value}")
        print(f"Input: {self.config.input_mode}")
        print(f"Response: {self.config.response_mode}")
        if self.config.enable_voice:
            print(f"Note: Audio responses enabled - you will hear Gemini speak")
        print(f"Compositional: {self.config.enable_compositional}")
        
        async with self.client.aio.live.connect(
            model=self.config.model,
            config=config
        ) as session:
            self.session = session
            self._session_active = True
            self._reconnect_attempts = 0  # Reset on successful connection
            self._last_successful_response = time.time()
            
            # Always initialize asyncio.Queue for audio output (prevents None checks)
            self.audio_output_queue = asyncio.Queue()
            
            # Start audio tasks now that session is active
            await self.start_audio_tasks()
            
            print(f"[CONNECTED] Gemini Live session established")
            print(f"Model: {self.config.model}")
            print(f"Tools: {len(robot_tools)} registered")
            print(f"Response Mode: {'Audio-only' if self.config.enable_voice else 'Text-only'}")
            
            # If we had a pending tool response, send it now
            if self._pending_tool_response:
                print("[RECONNECT] Sending pending tool response...")
                try:
                    await self.session.send_tool_response(
                        function_responses=[self._pending_tool_response]
                    )
                    self._pending_tool_response = None
                except Exception as e:
                    print(f"[WARNING] Could not send pending tool response: {e}")
            
            # Start response handler with monitoring
            response_task = asyncio.create_task(self._handle_responses())
            
            # Monitor connection health
            monitor_task = asyncio.create_task(self._monitor_connection_health())
            
            # Start audio player task if voice is enabled
            audio_player_task = None
            if self.config.enable_voice and self.audio_output_stream:
                audio_player_task = asyncio.create_task(self._audio_player())
                if self.config.enable_debug:
                    print("[DEBUG] Audio player task started")
            
            # Collect tasks to monitor
            tasks_to_monitor = [response_task, monitor_task]
            if audio_player_task:
                tasks_to_monitor.append(audio_player_task)
            
            try:
                # Wait for tasks
                done, pending = await asyncio.wait(
                    tasks_to_monitor,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Check if any completed task has an exception and re-raise it
                for task in done:
                    if task.exception() is not None:
                        # Cancel remaining tasks first
                        for pending_task in pending:
                            pending_task.cancel()
                            try:
                                await pending_task
                            except asyncio.CancelledError:
                                pass
                        # Now re-raise the exception to trigger reconnection
                        raise task.exception()
                
                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                        
            except KeyboardInterrupt:
                print("\n[INFO] Shutting down...")
            finally:
                self._session_active = False

    async def _handle_responses(self):
        """Enhanced response handler with error recovery"""
        if not self.session:
            return
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self._session_active:
            try:
                turn_text = ""
                
                async for response in self.session.receive():
                    try:
                        # Update last successful response time
                        self._last_successful_response = time.time()
                        consecutive_errors = 0  # Reset error counter
                        
                        # Text response accumulation
                        if response.text and not self._waiting_for_input:
                            turn_text += response.text
                        
                        # Audio response handling - Google's example shows response.data directly
                        if response.data:
                            # Audio comes as raw PCM data in response.data - streaming chunks
                            # Don't log every chunk as it floods the output
                            # Queue audio for playback - MUST await for asyncio.Queue
                            if self.audio_output_queue:
                                await self.audio_output_queue.put(response.data)
                        
                        # Tool invocation processing
                        if response.tool_call:
                            if turn_text:
                                print(f"Gemini: {turn_text}")
                                turn_text = ""
                            
                            if self.config.enable_debug:
                                print("[DEBUG] Tool call detected")
                            
                            # Tool execution with error handling
                            try:
                                await self._process_tool_calls(response.tool_call)
                            except Exception as e:
                                print(f"[ERROR] Tool call processing failed: {e}")
                                # Store for retry after reconnection if needed
                                if "1011" in str(e) or "internal" in str(e).lower():
                                    raise  # Re-raise to trigger reconnection
                        
                        # Server response processing
                        if response.server_content:
                            if response.server_content.model_turn and response.server_content.model_turn.parts:
                                for part in response.server_content.model_turn.parts:
                                    if part.function_call:
                                        if turn_text:
                                            print(f"Gemini: {turn_text}")
                                            turn_text = ""
                                        
                                        if self.config.enable_debug:
                                            print("[DEBUG] Function call in part")
                                        
                                        try:
                                            await self._process_single_function_call(part.function_call)
                                        except Exception as e:
                                            print(f"[ERROR] Function call failed: {e}")
                                            if "1011" in str(e):
                                                raise
                                    
                                    if part.executable_code:
                                        if self.config.enable_debug:
                                            print("[DEBUG] ExecutableCode (ignoring)")
                            
                            # Conversation turn completion
                            if response.server_content.turn_complete:
                                if turn_text:
                                    print(f"Gemini: {turn_text}")
                                    turn_text = ""
                                
                                if self.config.enable_debug:
                                    print("[DEBUG] Turn complete")
                                
                                if self.config.mode == OperationMode.TEXT:
                                    self._waiting_for_response = False
                                    
                    except Exception as e:
                        if self.config.enable_debug:
                            print(f"[ERROR] Processing response: {e}")
                        consecutive_errors += 1
                        
                        if consecutive_errors >= max_consecutive_errors:
                            print(f"[ERROR] Too many consecutive errors, triggering reconnection...")
                            raise ConnectionError("Too many consecutive response errors")
                            
            except asyncio.CancelledError:
                print("[INFO] Response handler cancelled")
                break
                
            except Exception as e:
                error_msg = str(e)
                
                # Connection error detection
                if any(err in error_msg.lower() for err in ['1011', 'internal error', 'closed', 'stream']):
                    print(f"[ERROR] Connection error detected: {error_msg}")
                    raise  # Re-raise to trigger reconnection
                else:
                    print(f"[ERROR] Response handler error: {e}")
                    if self.config.enable_debug:
                        traceback.print_exc()
                
                self._waiting_for_response = False
                await asyncio.sleep(1)

    async def _monitor_connection_health(self):
        """Monitor connection health and trigger reconnection if needed"""
        consecutive_timeouts = 0
        max_consecutive_timeouts = 3
        
        while self._session_active:
            try:
                await asyncio.sleep(5)  # 5-second health check interval
                
                # Response timeout detection
                time_since_last = time.time() - self._last_successful_response
                
                if time_since_last > self._connection_timeout:
                    consecutive_timeouts += 1
                    print(f"[WARNING] No responses for {time_since_last:.1f}s, connection may be dead (timeout {consecutive_timeouts}/{max_consecutive_timeouts})")
                    
                    # After multiple consecutive timeouts, force reconnection
                    if consecutive_timeouts >= max_consecutive_timeouts:
                        print(f"[ERROR] Connection appears dead after {consecutive_timeouts} consecutive timeouts")
                        raise ConnectionError(f"Connection timeout - no responses for {time_since_last:.1f}s")
                    
                    # Try sending a simple status request to test connection
                    # Skip this for audio mode as it interferes with audio flow
                    if self.config.input_mode != "audio":
                        try:
                            if self.session:
                                # Send a lightweight message to test connection
                                # Use send_client_content to avoid deprecation warning
                                await self.session.send_client_content(
                                    turns=[{
                                        'role': 'user',
                                        'parts': [types.Part(text=".")]
                                    }],
                                    turn_complete=True
                                )
                        except Exception as e:
                            print(f"[ERROR] Connection test failed: {e}")
                            raise ConnectionError("Connection health check failed")
                else:
                    # Reset counter if we're getting responses
                    consecutive_timeouts = 0
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ERROR] Health monitor error: {e}")
                raise
    
    async def _process_tool_calls(self, tool_call):
        """Process tool_call which contains function_calls array"""
        if not tool_call:
            return
        
        if hasattr(tool_call, 'function_calls'):
            for fc in tool_call.function_calls:
                await self._process_single_function_call(fc)

    async def _process_single_function_call(self, function_call):
        """Process a single function call"""
        if not function_call:
            return
        
        function_name = getattr(function_call, 'name', None)
        call_id = getattr(function_call, 'id', None)
        arguments = dict(function_call.args) if hasattr(function_call, 'args') else {}
        
        if not function_name:
            if self.config.enable_debug:
                print("[ERROR] Function call has no name")
            return
        
        print(f"-> Calling {function_name} with {arguments}")
        
        # Check robot connection for robot commands
        robot_required_commands = [
            "move_robot_joints", "move_robot_pose", "pick_up_object", 
            "place_object", "approach_object", "home_robot", "control_gripper",
            "search_for_object", "turn_to_face", "wave"
        ]
        
        if function_name in robot_required_commands:
            if not self._robot_checked:
                await self.get_robot_status()
            
            if not self._robot_connected:
                error_msg = "Robot is not connected. Please connect and power on the robot."
                print(f"[ERROR] {error_msg}")
                
                function_response = types.FunctionResponse(
                    id=call_id or function_name,
                    name=function_name,
                    response={"error": error_msg, "success": False}
                )
                await self.session.send_tool_response(function_responses=[function_response])
                return
        
        # Update movement state and streaming
        movement_commands = ["search_for_object", "approach_object", "turn_to_face"]
        stop_commands = ["stop_when_found", "stop_robot"]
        
        if function_name in movement_commands:
            self._movement_active = True
            await self._start_livestream()
        elif function_name in stop_commands:
            self._movement_active = False
            await self._stop_livestream()
        
        # Execute the function
        function_to_call = self.tool_registry.get(function_name)
        
        if function_to_call:
            try:
                # Handle async vs sync functions
                if asyncio.iscoroutinefunction(function_to_call):
                    # It's an async function, await it directly
                    result = await function_to_call(**arguments)
                else:
                    # It's a sync function - first try calling it normally
                    # to check if it returns a coroutine
                    result = function_to_call(**arguments)
                    if asyncio.iscoroutine(result):
                        # It returned a coroutine, await it
                        result = await result
                    else:
                        # It's a regular sync function that returned a value
                        # We already have the result, no need to call again
                        pass
                
                print(f"<- Result: {result}")
                
                function_response = types.FunctionResponse(
                    id=call_id or function_name,
                    name=function_name,
                    response={"result": str(result)}
                )
                
                # Try to send the response
                try:
                    await self.session.send_tool_response(function_responses=[function_response])
                except Exception as e:
                    if "1011" in str(e) or "internal" in str(e).lower():
                        # Store for retry after reconnection
                        self._pending_tool_response = function_response
                        print("[WARNING] Stored tool response for retry after reconnection")
                        raise
                
            except Exception as e:
                print(f"Error in {function_name}: {e}")
                if self.config.enable_debug:
                    import traceback
                    traceback.print_exc()
                
                function_response = types.FunctionResponse(
                    id=call_id or function_name,
                    name=function_name,
                    response={"error": str(e)}
                )
                await self.session.send_tool_response(function_responses=[function_response])
        else:
            print(f"[ERROR] Unknown function: {function_name}")
    
    # Streaming management
    async def _start_livestream(self):
        """Start livestreaming when robot is moving"""
        if not self._is_livestreaming:
            self._is_livestreaming = True
            self._last_position_snapshot_sent = False
            if self.config.enable_debug:
                print("Starting livestream (robot moving)")
    
    async def _stop_livestream(self):
        """Stop livestreaming when robot stops"""
        if self._is_livestreaming:
            self._is_livestreaming = False
            self._last_position_snapshot_sent = False
            if self.config.enable_debug:
                print("Stopping livestream (robot stationary)")
    
    def should_send_frame(self) -> bool:
        """Determine if a frame should be sent based on current state and mode"""
        current_time = time.time()
        
        # In autonomous mode with continuous coverage, always stream during movement
        if self.config.mode == OperationMode.AUTONOMOUS and self._movement_active:
            frame_interval = 1.0 / self.config.livestream_fps
            if current_time - self._last_snapshot_time >= frame_interval:
                self._last_snapshot_time = current_time
                return True
        
        # Standard streaming logic for text mode
        elif self._is_livestreaming:
            frame_interval = 1.0 / self.config.livestream_fps
            if current_time - self._last_snapshot_time >= frame_interval:
                self._last_snapshot_time = current_time
                return True
        else:
            # Send snapshot periodically when stationary
            if not self._last_position_snapshot_sent:
                self._last_snapshot_time = current_time
                self._last_position_snapshot_sent = True
                return True
        
        return False
    
    async def send_frame(self):
        """Send current camera frame to Gemini with search instructions if searching"""
        if not self.session or not self.vision_controller:
            return
        
        try:
            color_frame, _ = self.vision_controller.get_frames()
            if color_frame is None:
                return
            
            # Convert to PIL Image
            image = PIL.Image.fromarray(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))
            
            # Check if we're searching and should send a reminder
            # Only send reminders if we're actively moving and searching
            if self._current_search_target and self._movement_active:
                # Initialize frame counter if needed
                if not hasattr(self, '_frames_since_reminder'):
                    self._frames_since_reminder = 0
                
                self._frames_since_reminder += 1
                
                # Send reminder every 30 frames (about 3 seconds at 10fps)
                if self._frames_since_reminder >= 10:
                    # Send frame WITH text reminder - just like normal user messages
                    img_thumb = image.copy()
                    img_thumb.thumbnail([1024, 1024])
                    
                    import io
                    image_io = io.BytesIO()
                    img_thumb.save(image_io, format="jpeg")
                    image_bytes = image_io.getvalue()
                    
                    # Build message parts - EXACTLY like in _text_input_loop
                    turn_parts = []
                    
                    # Add image part
                    image_part = types.Part(
                        inline_data=types.Blob(
                            mime_type="image/jpeg",
                            data=image_bytes
                        )
                    )
                    turn_parts.append(image_part)
                    
                    # Add text part
                    reminder_text = f"Still searching for {self._current_search_target}. If you see it in this frame or upcoming ones, call stop_when_found() ONLY if >80% certain. You need to be certain it is visible in frame, otherwise do not stop."
                    text_part = types.Part(text=reminder_text)
                    turn_parts.append(text_part)
                    
                    # Send message with both image and text together
                    await self.session.send_client_content(
                        turns=[{'role': 'user', 'parts': turn_parts}],
                        turn_complete=True
                    )
                    
                    self._frames_since_reminder = 0
                    
                    if self.config.enable_debug:
                        print(f"[DEBUG] Sent search reminder with frame: {reminder_text}")
                else:
                    # Normal frame sending between reminders
                    await self.session.send_realtime_input(media=image)
                    
                    if self.config.enable_debug and not self._waiting_for_input:
                        print("[DEBUG] Frame sent")
            else:
                # Not searching, just send frame normally
                await self.session.send_realtime_input(media=image)
                
                if self.config.enable_debug and not self._waiting_for_input:
                    print("[DEBUG] Frame sent")
                
        except Exception as e:
            if self.config.enable_debug and not self._waiting_for_input:
                print(f"[ERROR] Failed to send frame: {e}")
    
    async def get_robot_status(self) -> Dict:
        """Get current robot status"""
        if self._robot_checked and not self._robot_connected:
            return {"status": "disconnected", "connected": False}
        
        try:
            pose = get_robot_pose()
            joints = get_robot_joint_angles()
            
            self._robot_checked = True
            self._robot_connected = True
            
            status = {
                "pose": pose,
                "joints": joints,
                "is_moving": self._movement_active,
                "connected": True
            }
            
            if self.robot_tools:
                tools_status = self.robot_tools.get_status()
                status.update(tools_status)
            
            return status
            
        except Exception as e:
            self._robot_checked = True
            self._robot_connected = False
            
            if self.config.enable_debug:
                print("Robot not connected - operating in vision-only mode")
            
            return {"status": "disconnected", "message": str(e), "connected": False}
        
    async def setup_audio(self):
        """Initialize audio components for push-to-talk and/or audio playback"""
        if self.config.input_mode != "audio" and not self.config.enable_voice:
            return
            
        try:
            self.pya = pyaudio.PyAudio()
            
            # Setup input stream if using audio input
            if self.config.input_mode == "audio":
                mic_info = self.pya.get_default_input_device_info()
                
                self.audio_stream = self.pya.open(
                    format=AUDIO_FORMAT,
                    channels=AUDIO_CHANNELS,
                    rate=AUDIO_RATE,
                    input=True,
                    input_device_index=mic_info["index"],
                    frames_per_buffer=AUDIO_CHUNK
                )
                
                print(f"Audio input initialized. Hold [{self.config.push_to_talk_key}] to speak")
                
                # NOTE: Audio sending task and keyboard listener will be started
                # in start_audio_tasks() after the session is active
            
            # Setup output stream if using voice responses
            if self.config.enable_voice:
                print(f"[DEBUG] Initializing audio output (enable_voice={self.config.enable_voice})")
                self.audio_output_stream = self.pya.open(
                    format=AUDIO_FORMAT,
                    channels=AUDIO_CHANNELS,
                    rate=24000,  # Gemini outputs at 24kHz
                    output=True,
                    frames_per_buffer=AUDIO_CHUNK
                )
                
                print("[OK] Audio output initialized for voice responses")
                
                # Note: Audio playback task is started in run_session()
            
        except Exception as e:
            print(f"Failed to initialize audio: {e}")
            if self.config.input_mode == "audio":
                self.config.input_mode = "text"  # Fallback to text
    
    async def start_audio_tasks(self):
        """Start audio tasks after session is active"""
        if self.config.input_mode == "audio":
            # Start audio sending task
            self._audio_task = asyncio.create_task(self._audio_sender())
            
            # Setup keyboard listener in a thread
            threading.Thread(target=self._keyboard_listener, daemon=True).start()
            
            if self.config.enable_debug:
                print("[DEBUG] Audio tasks started")
    
    def _keyboard_listener(self):
        """Thread to listen for push-to-talk key"""
        while self._session_active:
            try:
                if keyboard.is_pressed(self.config.push_to_talk_key):
                    if not self._recording:
                        self._recording = True
                        
                        # Capture and immediately send current frame for visual context
                        if self.vision_controller:
                            color_frame, _ = self.vision_controller.get_frames()
                            if color_frame is not None:
                                self._audio_frame_for_context = color_frame.copy()
                                # Queue the frame to be sent immediately when recording starts
                                self.audio_queue.put({"send_frame": True})
                                if self.config.enable_debug:
                                    print("[DEBUG] Captured frame for audio context")
                        
                        print("[Recording...]", end="\r")
                        threading.Thread(target=self._record_audio, daemon=True).start()
                else:
                    if self._recording:
                        self._recording = False
                        print("[Processing...] ", end="\r")
                        # Signal end of recording
                        self.audio_queue.put({"end_of_recording": True})
                        
                time.sleep(0.01)  # Small delay to prevent CPU spinning
                
            except Exception as e:
                if self.config.enable_debug:
                    print(f"Keyboard listener error: {e}")
    
    def _record_audio(self):
        """Record audio while key is held"""
        while self._recording and self.audio_stream:
            try:
                data = self.audio_stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                # Queue audio for real-time streaming (following Google's approach)
                self.audio_queue.put({"data": data, "mime_type": "audio/pcm"})
            except Exception as e:
                if self.config.enable_debug:
                    print(f"Audio recording error: {e}")
                break
    
    async def _audio_sender(self):
        """Send audio data to Gemini following Google's example approach"""
        while self._session_active:
            try:
                # Use timeout to prevent blocking forever
                audio_data = await asyncio.to_thread(
                    self.audio_queue.get, timeout=0.1
                )
                
                if self.session:
                    if "send_frame" in audio_data:
                        # Send the captured frame at the start of recording
                        if self._audio_frame_for_context is not None:
                            frame_rgb = cv2.cvtColor(self._audio_frame_for_context, cv2.COLOR_BGR2RGB)
                            img = PIL.Image.fromarray(frame_rgb)
                            img.thumbnail([1024, 1024])
                            
                            # Send image using send_realtime_input
                            await self.session.send_realtime_input(media=img)
                            
                            if self.config.enable_debug:
                                print(f"[DEBUG] Sent frame at start of audio recording")
                            
                            # Don't clear the frame yet, we might need it
                        
                    elif "end_of_recording" in audio_data:
                        # Recording finished - send silence then signal end of turn
                        if self.config.enable_debug:
                            print("[DEBUG] Audio recording ended, signaling end of turn")
                        
                        # Send a small silence buffer to help Gemini detect end of speech
                        # Increase to 500ms for better detection
                        silence_duration = 0.5  # 500ms of silence
                        silence_samples = int(16000 * silence_duration)  # 16kHz sample rate
                        silence_data = b'\x00' * (silence_samples * 2)  # 2 bytes per sample (16-bit)
                        
                        await self.session.send_realtime_input(
                            audio=types.Blob(
                                data=silence_data,
                                mime_type="audio/pcm"
                            )
                        )
                        
                        # Small delay to ensure all audio has been transmitted
                        await asyncio.sleep(0.1)
                        
                        # Clear frame buffer
                        self._audio_frame_for_context = None
                        
                        # Do NOT send explicit turn completion for audio
                        # Google's example shows that audio should rely on natural turn detection
                        # The silence buffer we sent above is sufficient for Gemini to detect end of speech
                        if self.config.enable_debug:
                            print("[DEBUG] Audio recording ended, relying on silence for turn detection")
                        
                    elif "data" in audio_data:
                        # Send audio chunk in real-time (following Google's approach)
                        audio_bytes = audio_data["data"]
                        
                        # Send using send_realtime_input for real-time streaming
                        await self.session.send_realtime_input(
                            audio=types.Blob(
                                data=audio_bytes,
                                mime_type="audio/pcm"
                            )
                        )
                        
            except queue.Empty:
                continue
            except Exception as e:
                if self.config.enable_debug:
                    print(f"Audio sender error: {e}")
                await asyncio.sleep(0.1)
    
    async def _audio_player(self):
        """Play audio responses from Gemini"""
        while self._session_active:
            try:
                # Get audio data from queue - blocking call
                audio_bytes = await self.audio_output_queue.get()
                
                if self.audio_output_stream and audio_bytes:
                    # Audio from Gemini is 24kHz, 16-bit PCM
                    # Write directly to output stream (following Google's example)
                    await asyncio.to_thread(self.audio_output_stream.write, audio_bytes)
                    
            except Exception as e:
                if self.config.enable_debug:
                    print(f"[DEBUG] Audio playback error: {e}")
                await asyncio.sleep(0.1)
        
    async def _cleanup_session(self):
        """Clean up current session before reconnection"""
        try:
            # Stop recording if in progress
            self._recording = False
            
            # Clear audio context
            self._audio_frame_for_context = None
            
            # Clear audio queues
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except:
                    pass
            
            # Clear output queue if it exists (asyncio.Queue)
            if self.audio_output_queue:
                if hasattr(self.audio_output_queue, 'get_nowait'):
                    while not self.audio_output_queue.empty():
                        try:
                            self.audio_output_queue.get_nowait()
                        except:
                            pass
            
            if self.session:
                try:
                    await self.session.close()
                except:
                    pass  # Ignore errors during cleanup
                self.session = None
            
            self._session_active = False
            
            # Don't close vision/robot components - keep them for reconnection
            print("[CLEANUP] Session cleaned up, audio buffers cleared, components preserved")
            
        except Exception as e:
            print(f"[WARNING] Cleanup error: {e}")
    
    async def cleanup(self):
        """Clean up all resources completely"""
        try:
            # Clean up session first
            await self._cleanup_session()
            
            # Clean up audio resources
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            if self.audio_output_stream:
                self.audio_output_stream.stop_stream()
                self.audio_output_stream.close()
            if self.pya:
                self.pya.terminate()
            
            # Cancel audio tasks
            if self._audio_task:
                self._audio_task.cancel()
                try:
                    await self._audio_task
                except asyncio.CancelledError:
                    pass
            if self._audio_playback_task:
                self._audio_playback_task.cancel()
                try:
                    await self._audio_playback_task
                except asyncio.CancelledError:
                    pass

            # Clean up vision
            if self.vision_controller:
                self.vision_controller.stop_display_thread()
                self.vision_controller.stop_camera()
            
            print("Cleanup complete")
        except Exception as e:
            print(f"Cleanup error: {e}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class GeminiLiveApp:
    """Main application coordinating Gemini Live with robot"""
    
    def __init__(self, mode: str = "text", input_mode: str = "text", response_mode: str = "text"):
        self.config = GeminiConfig.from_mode(mode, input_mode, response_mode)
        self.session_manager = GeminiLiveSession(self.config)
        self.running = False
    
    async def initialize(self):
        """Initialize all components"""
        print(f"\n{'='*60}")
        print(f"PAROL6 Gemini Live Controller")
        print(f"Mode: {self.config.mode.value.upper()}")
        print(f"Input: {self.config.input_mode} | Response: {self.config.response_mode}")
        if self.config.mode == OperationMode.AUTONOMOUS:
            print("Compositional function calling enabled")
            print(f"Turn coverage: {self.config.turn_coverage}")
        print(f"{'='*60}\n")
        
        success = self.session_manager.initialize_components()
        if not success:
            return False
        
        print("\nSystem ready")
        if self.config.mode == OperationMode.AUTONOMOUS:
            print("Autonomous operation enabled - multi-step sequences supported")
        else:
            print("Text mode - step-by-step control")
        print("-" * 60)
        
        return True
    
    async def run(self):
        """Main run loop"""
        self.running = True
        
        # Setup audio BEFORE starting session (for input or output)
        # This ensures audio output stream is ready when session starts
        if self.config.input_mode == "audio" or self.config.enable_voice:
            await self.session_manager.setup_audio()
        
        # Start the session
        session_task = asyncio.create_task(self.session_manager.run_session())
        
        await asyncio.sleep(2)
        
        # Start frame sending loop
        frame_task = asyncio.create_task(self._frame_loop())
        
        try:
            if self.config.mode == OperationMode.AUTONOMOUS:
                print("\nAUTONOMOUS MODE ACTIVE")
                if self.config.input_mode == "audio":
                    print(f"PUSH-TO-TALK: Hold [{self.config.push_to_talk_key}] and speak")
                    print("Examples while holding key:")
                    print("  - 'Pick up the red block and place it on the table'")
                    print("  - 'Find and grab the blue cube'")
                else:
                    print("TEXT INPUT MODE")
                    print("Type commands:")
                    print("  - Pick up the red block and place it on the table")
                    print("  - Find and grab the blue cube")
                print("\nPress Ctrl+C to exit\n")
            else:
                print("\nTEXT MODE ACTIVE")
                print("Examples:")
                print("  - pick up the red block")
                print("  - search for object")
                print("  - wave")
                print("\nType 'quit' to exit\n")
            
            # Choose input method based on config
            if self.config.input_mode == "audio":
                # In audio mode, just wait for session to complete
                await session_task
            else:
                # Text input mode
                input_task = asyncio.create_task(self._text_input_loop())
                
                # Wait for either to complete
                done, pending = await asyncio.wait(
                    [session_task, input_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in pending:
                    task.cancel()
                    
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        finally:
            self.running = False
            self.session_manager._session_active = False
            frame_task.cancel()
            await self.session_manager.cleanup()
    
    async def _frame_loop(self):
        """Smart frame sending loop"""
        while self.running:
            try:
                # In autonomous mode, continuously stream during movement
                if self.config.mode == OperationMode.AUTONOMOUS and self.session_manager._movement_active:
                    if self.session_manager.session:
                        await self.session_manager.send_frame()
                    await asyncio.sleep(1.0 / self.config.livestream_fps)
                    
                # Standard frame sending logic
                elif self.session_manager.session and self.session_manager.should_send_frame():
                    await self.session_manager.send_frame()
                    await asyncio.sleep(0.05)
                    
                else:
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                if self.config.enable_debug:
                    print(f"Frame loop error: {e}")
                await asyncio.sleep(1)
    
    async def _text_input_loop(self):
        """Send messages with mode-specific behavior"""
        while self.running:
            try:
                # In autonomous mode, don't wait for response completion
                if self.config.mode == OperationMode.AUTONOMOUS:
                    user_input = await asyncio.to_thread(input, "Command: ")
                else:
                    # In text mode, wait for response before next prompt
                    if not self.session_manager._waiting_for_response:
                        user_input = await asyncio.to_thread(input, "You: ")
                    else:
                        await asyncio.sleep(0.1)
                        continue
                
                if user_input.lower() in ['quit', 'exit', 'stop']:
                    self.running = False
                    self.session_manager._session_active = False
                    break
                
                if not user_input.strip():
                    continue
                
                if self.session_manager.session:
                    # Set waiting flag in text mode
                    if self.config.mode == OperationMode.TEXT:
                        self.session_manager._waiting_for_response = True
                    
                    # Build message parts
                    turn_parts = []
                    
                    # Add image if available
                    if self.session_manager.vision_controller:
                        color_frame, _ = self.session_manager.vision_controller.get_frames()
                        if color_frame is not None:
                            frame_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                            img = PIL.Image.fromarray(frame_rgb)
                            img.thumbnail([1024, 1024])
                            
                            import io
                            image_io = io.BytesIO()
                            img.save(image_io, format="jpeg")
                            image_bytes = image_io.getvalue()
                            
                            image_part = types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/jpeg",
                                    data=image_bytes
                                )
                            )
                            turn_parts.append(image_part)
                            
                            if self.session_manager.config.enable_debug:
                                print(f"[DEBUG] Added image ({len(image_bytes)} bytes)")
                    
                    # Add text part
                    text_part = types.Part(text=user_input)
                    turn_parts.append(text_part)
                    
                    # Send message
                    await self.session_manager.session.send_client_content(
                        turns=[{'role': 'user', 'parts': turn_parts}],
                        turn_complete=True
                    )
                    
                    # In text mode, wait for response
                    if self.config.mode == OperationMode.TEXT:
                        while self.session_manager._waiting_for_response and self.running:
                            await asyncio.sleep(0.1)
                        
                else:
                    print("Session not active yet. Please wait...")
                    
            except EOFError:
                break
            except Exception as e:
                if self.session_manager.config.enable_debug:
                    print(f"Input error: {e}")
                    traceback.print_exc()
                self.session_manager._waiting_for_response = False
                await asyncio.sleep(0.1)

# Additional helper to gracefully restart the entire app if needed
async def run_with_auto_restart(mode: str = "text", input_mode: str = "text", response_mode: str = "text"):
    """Run the app with automatic restart on fatal errors"""
    restart_attempts = 0
    max_restarts = 3
    
    while restart_attempts < max_restarts:
        try:
            print(f"\n[START] Starting Gemini Live App (attempt {restart_attempts + 1}/{max_restarts})")
            
            app = GeminiLiveApp(mode=mode, input_mode=input_mode, response_mode=response_mode)
            
            if not await app.initialize():
                print("[ERROR] Failed to initialize")
                restart_attempts += 1
                await asyncio.sleep(5)
                continue
            
            await app.run()
            break  # Normal exit
            
        except Exception as e:
            print(f"\n[FATAL] Application crashed: {e}")
            restart_attempts += 1
            
            if restart_attempts < max_restarts:
                print(f"[RESTART] Restarting application in 10 seconds...")
                await asyncio.sleep(10)
            else:
                print("[FATAL] Max restart attempts reached. Please check the system.")
                break
    
    print("[END] Application terminated")

# ============================================================================
# ENTRY POINT
# ============================================================================

async def main():
    """Enhanced main entry point with auto-restart capability"""
    parser = argparse.ArgumentParser(description='PAROL6 Gemini Live Controller')
    parser.add_argument(
        '--mode',
        choices=['text', 'autonomous'],
        default='text',
        help='Operation mode: text (step-by-step) or autonomous (compositional)'
    )
    parser.add_argument(
        '--input',
        choices=['text', 'audio'],
        default='text',
        help='Input mode: text (keyboard) or audio (push-to-talk). Default: text'
    )
    parser.add_argument(
        '--response',
        choices=['text', 'audio'],
        default=None,  # Will be determined based on input mode
        help='Response mode: text (terminal) or audio (voice). Default: matches input mode (audio→audio, text→text)'
    )
    parser.add_argument(
        '--ptt-key',
        default='space',
        help='Push-to-talk key for audio input (default: space). Options: space, ctrl, shift, tab, etc.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    parser.add_argument(
        '--no-restart',
        action='store_true',
        help='Disable automatic restart on failures'
    )
    
    args = parser.parse_args()
    
    # Implement smart defaults for response mode
    if args.response is None:
        # If no response mode specified, default based on input mode
        if args.input == "audio":
            response_mode = "audio"  # Audio input defaults to audio response
            if args.debug:
                print(f"[DEBUG] Auto-setting response mode to 'audio' (matches input)")
        else:
            response_mode = "text"   # Text input defaults to text response
            if args.debug:
                print(f"[DEBUG] Auto-setting response mode to 'text' (matches input)")
    else:
        response_mode = args.response  # Use explicit override
        if args.debug:
            print(f"[DEBUG] Using explicit response mode: {response_mode}")
    
    # If auto-restart is disabled, run normally
    if args.no_restart:
        app = GeminiLiveApp(mode=args.mode, input_mode=args.input, response_mode=response_mode)
        
        # Override settings if specified
        if args.ptt_key:
            app.config.push_to_talk_key = args.ptt_key
        if args.debug:
            app.config.enable_debug = True
        
        if await app.initialize():
            await app.run()
        else:
            print("Failed to initialize system")
            return 1
        return 0
    
    # Otherwise, run with auto-restart capability
    restart_attempts = 0
    max_restarts = 3
    
    while restart_attempts < max_restarts:
        try:
            print(f"\n{'='*60}")
            if restart_attempts > 0:
                print(f"[RESTART] Attempt {restart_attempts + 1}/{max_restarts}")
            print(f"{'='*60}\n")
            
            # Create and configure app
            app = GeminiLiveApp(mode=args.mode, input_mode=args.input, response_mode=response_mode)
            
            # Override settings if specified
            if args.ptt_key:
                app.config.push_to_talk_key = args.ptt_key
            if args.debug:
                app.config.enable_debug = True
            
            # Initialize components
            if not await app.initialize():
                print("[ERROR] Failed to initialize")
                restart_attempts += 1
                if restart_attempts < max_restarts:
                    print(f"[RETRY] Waiting 5 seconds before retry...")
                    await asyncio.sleep(5)
                continue
            
            # Run the application
            await app.run()
            
            # If we get here, it's a normal exit
            print("\n[INFO] Normal exit")
            break
            
        except KeyboardInterrupt:
            print("\n[INFO] User interrupted")
            break
            
        except Exception as e:
            print(f"\n[ERROR] Application crashed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            
            restart_attempts += 1
            
            if restart_attempts < max_restarts:
                wait_time = 10 * restart_attempts  # 10s, 20s, 30s
                print(f"[RESTART] Restarting application in {wait_time} seconds...")
                print("[INFO] Press Ctrl+C to cancel restart")
                
                try:
                    await asyncio.sleep(wait_time)
                except KeyboardInterrupt:
                    print("\n[INFO] Restart cancelled by user")
                    break
            else:
                print("[FATAL] Max restart attempts reached")
                print("[INFO] Please check the system and restart manually")
                return 1
    
    return 0

if __name__ == "__main__":
    # Suppress the "Warning: there are non-text parts" messages from Gemini API
    # These are informational warnings about multi-modal responses (audio/video/text)
    warnings.filterwarnings("ignore", message=".*non-text parts.*")
    warnings.filterwarnings("ignore", message=".*non-data parts.*")
    warnings.filterwarnings("ignore", message=".*non text parts.*")
    
    # Calibrate gripper on startup (optional)
    try:
        from Headless.robot_api import control_electric_gripper
        control_electric_gripper("calibrate")
        print(" Gripper calibrated")

        from supplementary_functions import move_to_standard_position
        move_to_standard_position()
        print(" Moved to home position")

    except Exception as e:
        print(f"[WARNING] Gripper calibration skipped: {e}")
    
    # Run with proper exit code handling
    try:
        import sys
        exit_code = asyncio.run(main())
        sys.exit(exit_code if exit_code else 0)
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL] Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

###########################################################################
# USAGE INSTRUCTIONS
###########################################################################

# Basic Configurations

    # Default Behavior
""" 
# Default - text input with text responses
python gemini.py

# Audio input with audio responses (default for audio)
python gemini.py --input audio

# Text input with text responses (explicit)
python gemini.py --input text --response text
 """

    # Mixed Input/Output Modes
""" 
# Speak to Gemini, see text responses
python gemini.py --input audio --response text

# Type to Gemini, hear voice responses
python gemini.py --input text --response audio

# Full voice conversation (explicit)
python gemini.py --input audio --response audio
 """

    # Autonomous Mode Examples
""" 
# Autonomous with text I/O
python gemini.py --mode autonomous

# Autonomous with voice I/O (defaults to audio responses)
python gemini.py --mode autonomous --input audio

# Autonomous with voice input but text responses
python gemini.py --mode autonomous --input audio --response text

# Custom push-to-talk key
python gemini.py --mode autonomous --input audio --ptt-key ctrl
python gemini.py --mode autonomous --input audio --ptt-key shift

# With debug output
python gemini.py --mode autonomous --input audio --debug
 """

# Complete Parameter Reference
""" 
python gemini.py [OPTIONS]

Options:
  --mode {text,autonomous}
      text: Step-by-step execution, waits for input between functions
      autonomous: Compositional function calling, executes full sequences
      Default: text

  --input {text,audio}  
      text: Keyboard input
      audio: Push-to-talk voice input
      Default: text

  --response {text,audio}
      text: Terminal text output
      audio: Voice/speech output
      Default: Matches input mode (audio→audio, text→text)

  --ptt-key KEY
      Push-to-talk key for audio input
      Options: space, ctrl, shift, alt, tab, f1-f12, etc.
      Default: space

  --debug
      Enable debug output and verbose logging
      Default: False

  --no-restart
      Disable automatic restart on connection failures
      Default: False (auto-restart enabled)
 """

# Common Use Cases
""" 
# Testing individual functions
python gemini.py --mode text

# Full voice conversation (speak and hear responses)
python gemini.py --mode autonomous --input audio

# Voice input with text output (good for debugging)
python gemini.py --mode autonomous --input audio --response text

# Text input with voice output (accessibility)
python gemini.py --mode autonomous --response audio

# Debugging with full visibility
python gemini.py --mode autonomous --input audio --response text --debug

# Voice control with different PTT key
python gemini.py --mode autonomous --input audio --ptt-key ctrl

# Development/testing with all debug info
python gemini.py --mode text --debug """
