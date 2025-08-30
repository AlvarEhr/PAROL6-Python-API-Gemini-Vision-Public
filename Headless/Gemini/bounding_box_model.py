# Fixed version of get_boxes_from_pro_model function
import os
import json
import cv2
import numpy as np
import PIL.Image
from io import BytesIO
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig
from typing import Union, List, Dict, Optional

def parse_json(json_output: str) -> str:
    """Selects text between ```json and ```."""
    if "```json" in json_output:
        try:
            json_output = json_output.split("```json", 1)[1].split("```")[0]
        except IndexError:
            pass
    return json_output.strip()

def get_boxes_from_pro_model(image_input: Union[str, np.ndarray], prompt: str) -> Optional[List[Dict]]:
    """
    Get bounding boxes from Gemini Pro using either an image path or numpy array.
    
    Args:
        image_input: Either a file path (str) or numpy array (BGR format from OpenCV)
        prompt: Description of what to detect
        
    Returns:
        List of bounding boxes with format [{'box': [x1, y1, x2, y2], 'label': 'object'}, ...]
        or None if detection fails
    """
    print(f"\n--- Pro Model Handler ---")
    
    try:
        # Handle both file paths and numpy arrays
        if isinstance(image_input, str):
            # Load from file
            print(f"Loading image from file: {image_input}")
            with open(image_input, 'rb') as f:
                image_bytes = f.read()
            # Also load as PIL for size information
            pil_image = PIL.Image.open(image_input)
            
        elif isinstance(image_input, np.ndarray):
            # Convert numpy array to bytes
            print(f"Converting numpy array to bytes")
            # Assume the numpy array is in BGR format (from OpenCV)
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                # Convert BGR to RGB
                rgb_array = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            else:
                rgb_array = image_input
            
            # Convert to PIL Image
            pil_image = PIL.Image.fromarray(rgb_array)
            
            # Convert PIL Image to JPEG bytes
            buffer = BytesIO()
            pil_image.save(buffer, format='JPEG', quality=95)
            image_bytes = buffer.getvalue()
            
        else:
            print(f"Error: Unsupported image input type: {type(image_input)}")
            return None
        
        # Get original dimensions for coordinate conversion
        width, height = pil_image.size
        print(f"Image dimensions: {width}x{height}")
        
        # Create image part using the proper Gemini API method
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg'
        )
        
        # Initialize Gemini client
        client = genai.Client()
        
        # Simplified system instructions - Gemini already knows the box_2d format
        bounding_box_system_instructions = """Return bounding boxes as a JSON array with labels. 
        Find ONLY the single object that best matches the given description.
        If multiple similar objects exist, choose the most prominent or centered one.
        Never return masks or code fencing. 
        Return only the JSON array, no other text."""
        
        config = GenerateContentConfig(
            system_instruction=bounding_box_system_instructions,
            temperature=0.3,  # Lower temperature for more consistent detection
            thinking_config = types.ThinkingConfig(
            thinking_budget=128,
            ),
        )
        
        try:
            print(f"Sending request to Gemini Pro for: {prompt}")
            
            # Properly format the request with the image part
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    f"Find the object that best matches this description: {prompt}",
                    image_part
                ],
                config=config,
            )
            
            print("Received response from Gemini Pro.")
            
            # Parse the response
            cleaned_json_str = parse_json(response.text)
            if not cleaned_json_str:
                cleaned_json_str = response.text.strip()
                
            bounding_boxes = json.loads(cleaned_json_str)
            
            if not bounding_boxes:
                print("No objects detected")
                return None
            
            # Process the bounding boxes using Gemini's trained format
            result_boxes = []
            for bbox in bounding_boxes:
                # Gemini automatically uses "box_2d" format with normalized 0-1000 coordinates
                # IMPORTANT: The format is [y_min, x_min, y_max, x_max] - y coordinates come FIRST!
                # This is contrary to common usage where x typically comes first
                box_data = bbox.get("box_2d") or bbox.get("box")
                label = bbox.get("label", "detection")
                confidence = bbox.get("confidence", 0.5)
                
                if not isinstance(box_data, list) or len(box_data) != 4:
                    continue
                
                # Unpack in Gemini's order: y coordinates first, then x coordinates
                y_min_norm, x_min_norm, y_max_norm, x_max_norm = box_data
                
                # Convert from normalized (0-1000) to pixel coordinates
                x1 = int(x_min_norm / 1000 * width)
                y1 = int(y_min_norm / 1000 * height)
                x2 = int(x_max_norm / 1000 * width)
                y2 = int(y_max_norm / 1000 * height)
                
                # Validate coordinates
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))
                
                # Ensure x2 > x1 and y2 > y1
                if x2 <= x1 or y2 <= y1:
                    print(f"Invalid box coordinates: [{x1}, {y1}, {x2}, {y2}]")
                    continue
                
                result_boxes.append({
                    'box': [x1, y1, x2, y2],  # Return in standard [x1, y1, x2, y2] format for OpenCV
                    'label': label,
                    'confidence': confidence
                })
            
            # If multiple boxes returned despite instructions, take the one with highest confidence
            if len(result_boxes) > 1:
                print(f"Multiple boxes returned ({len(result_boxes)}), selecting best match")
                result_boxes = sorted(result_boxes, key=lambda x: x.get('confidence', 0), reverse=True)
                result_boxes = [result_boxes[0]]  # Take only the best one
            
            print(f"Detected: {result_boxes[0]['label'] if result_boxes else 'nothing'}")
            return result_boxes if result_boxes else None
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from Pro model: {e}")
            print(f"Raw response: {response.text[:500]}...")  # Show first 500 chars
            return None
        except Exception as e:
            print(f"Error during Gemini Pro call: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"Error in get_boxes_from_pro_model: {e}")
        import traceback
        traceback.print_exc()
        return None


# Keep the original plotting function for visualization if needed
def plot_bounding_boxes_cv(im_cv: np.ndarray, bboxes: List[Dict]) -> np.ndarray:
    """Draws bounding boxes on an OpenCV image."""
    annotated_frame = im_cv.copy()
    
    if not bboxes:
        return annotated_frame

    # Export the main functions
    __all__ = ['get_boxes_from_pro_model', 'plot_bounding_boxes_cv', 'parse_json']
        
    for bbox_data in bboxes:
        x1, y1, x2, y2 = bbox_data['box']
        label = bbox_data.get('label', 'detection')
        confidence = bbox_data.get('confidence', 0)
        
        # Draw rectangle
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label with confidence
        label_text = f"{label} ({confidence:.2f})" if confidence > 0 else label
        cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return annotated_frame