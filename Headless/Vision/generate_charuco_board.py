import cv2
import numpy as np
import sys

# --- Configuration ---
# NOTE: The default values have been adjusted to ensure the board fits on A4 paper.
SQUARES_X = 7          # Number of squares in X direction
SQUARES_Y = 5          # Number of squares in Y direction
SQUARE_LENGTH = 0.025  # Length of one square side in meters (25mm)
MARKER_LENGTH = 0.015  # Length of the ArUco marker side in meters (15mm)
FILENAME = "charuco_board.png"

# ArUco dictionary
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# --- Board Creation ---
def create_charuco_board():
    """
    Generates and saves a ChArUco board image.
    """
    # Create the board object using the modern OpenCV syntax
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        ARUCO_DICT
    )

    # --- Image Generation ---
    # Define A4 paper size in mm and pixels (at 300 DPI)
    img_width_mm = 210
    img_height_mm = 297
    dpi = 300
    img_width_px = int(img_width_mm * dpi / 25.4)
    img_height_px = int(img_height_mm * dpi / 25.4)

    # Calculate the total size of the board in meters
    board_width_m = SQUARES_X * SQUARE_LENGTH
    board_height_m = SQUARES_Y * SQUARE_LENGTH
    
    # NEW: Add a check to ensure the board fits on the page
    if board_width_m > (img_width_mm / 1000) or board_height_m > (img_height_mm / 1000):
        print("--- ERROR: Board dimensions are larger than A4 paper ---")
        print(f"Board size: {board_width_m*1000:.0f}mm x {board_height_m*1000:.0f}mm")
        print(f"A4 paper size: {img_width_mm}mm x {img_height_mm}mm")
        print("Please reduce SQUARE_LENGTH or the number of squares (SQUARES_X/SQUARES_Y).")
        sys.exit()

    # Create an empty image (white background) for the A4 page
    img = np.full((img_height_px, img_width_px), 255, dtype=np.uint8)

    # Calculate the size of the board in pixels to draw it
    board_width_px = int(board_width_m * dpi / 0.0254)
    board_height_px = int(board_height_m * dpi / 0.0254)
    
    # Generate an image of the board itself
    board_img = board.generateImage((board_width_px, board_height_px))
    
    # Calculate top-left corner to paste the board image, centering it
    x_offset = (img_width_px - board_width_px) // 2
    y_offset = (img_height_px - board_height_px) // 2

    # Place the board image onto the A4 canvas
    img[y_offset:y_offset + board_height_px, x_offset:x_offset + board_width_px] = board_img

    # --- Save the image ---
    cv2.imwrite(FILENAME, img)
    print(f"Successfully created ChArUco board image: '{FILENAME}'")
    print(f"Board dimensions: {SQUARES_X}x{SQUARES_Y} squares")
    print(f"Square size: {SQUARE_LENGTH*1000} mm, Marker size: {MARKER_LENGTH*1000} mm")
    print("Please print this file and mount it on a rigid, flat surface.")

if __name__ == "__main__":
    create_charuco_board()
