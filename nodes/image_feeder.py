import os
from typing import List, Dict, Tuple

import torch
from PIL import Image

from ..core.image_processor import ImageProcessor


class ImageFeeder:
    """ComfyUI node for feeding images from a directory with frame limiting capabilities."""

    def __init__(self):
        self.current_dir = None
        self.image_files = []
        self.image_processor = ImageProcessor()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "current_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1
                }),
                "number_of_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "description": "Number of frames to process (0 = all frames)"
                }),
                "skip_first_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "description": "Number of frames to skip from the beginning"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT")
    RETURN_NAMES = ("image", "fp_pipe")
    FUNCTION = "feed_images"
    CATEGORY = "Face Processor/Image"

    def _load_image(self, image_path: str) -> torch.Tensor:
        """
        Load an image from path and convert it to ComfyUI format

        Args:
            image_path: Path to the image file

        Returns:
            torch.Tensor: Image in ComfyUI format (B,H,W,C) normalized to 0-1 range
        """
        try:
            image = Image.open(image_path)

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Use ImageProcessor to convert to tensor
            return self.image_processor.pil_to_tensor(image)

        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None

    def _scan_directory(self, directory: str) -> List[str]:
        """
        Scan directory for image files

        Args:
            directory: Path to directory to scan

        Returns:
            List of image file paths
        """
        # List of supported image extensions
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

        image_files = []

        try:
            # Get absolute path
            abs_dir = os.path.abspath(directory)

            if not os.path.exists(abs_dir):
                print(f"Directory not found: {abs_dir}")
                return image_files

            # Scan directory for image files
            print(f"Scanning directory: {abs_dir}")

            for file in os.listdir(abs_dir):
                # Check file extension
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    full_path = os.path.join(abs_dir, file)
                    image_files.append(full_path)

            # Sort files for consistent ordering
            image_files.sort()

            print(f"Found {len(image_files)} image files")

        except Exception as e:
            print(f"Error scanning directory {directory}: {str(e)}")

        return image_files

    def feed_images(self, directory: str, current_frame: int, number_of_frames: int = 0, skip_first_frames: int = 0) -> \
    Tuple[torch.Tensor, Dict]:
        """
        Main processing function

        Args:
            directory: Path to images directory
            current_frame: Absolute index of the frame to return (index in the file list)
            number_of_frames: Number of frames to include in fp_pipe (0 = all frames)
            skip_first_frames: Number of frames to skip from the beginning in fp_pipe

        Returns:
            Tuple containing:
            - Selected frame as tensor (based on current_frame)
            - Dictionary with fp_pipe structure (based on skip_first_frames and number_of_frames)
        """
        # Check if directory changed
        if directory != self.current_dir:
            print(f"New directory detected, scanning: {directory}")
            self.image_files = self._scan_directory(directory)
            self.current_dir = directory

        # Handle empty directory case
        if not self.image_files:
            print("No images found in directory")
            # Return empty image and data
            empty_image = torch.zeros((1, 64, 64, 3))
            return empty_image, {"frames": {}}

        # PART 1: Handle current_frame for image output
        total_files = len(self.image_files)

        # Validate current_frame is within range
        if current_frame < 0:
            current_frame = 0
            print(f"Current frame index adjusted: {current_frame} (min value)")
        elif current_frame >= total_files:
            current_frame = total_files - 1
            print(f"Current frame index adjusted: {current_frame} (max value)")

        # Load the selected frame based on current_frame
        selected_frame = self._load_image(self.image_files[current_frame])

        if selected_frame is None:
            print(f"Failed to load frame at index {current_frame}")
            selected_frame = torch.zeros((1, 64, 64, 3))

        # PART 2: Handle the fp_pipe creation based on skip_first_frames and number_of_frames
        # Validate skip_first_frames
        if skip_first_frames < 0:
            skip_first_frames = 0
        elif skip_first_frames >= total_files:
            skip_first_frames = 0
            print(f"Skip first frames reset to 0 (was out of range)")

        # Determine range of frames to include in fp_pipe
        if number_of_frames <= 0:
            # Include all remaining frames
            end_frame = total_files
        else:
            # Include limited number of frames
            end_frame = min(skip_first_frames + number_of_frames, total_files)

        # Build the frames dictionary
        frames_dict = {}
        for i, abs_idx in enumerate(range(skip_first_frames, end_frame)):
            frame_key = f"frame_{i}"
            frames_dict[frame_key] = {
                "original_frame_index": abs_idx,
                "original_image_path": self.image_files[abs_idx]
            }

        # Create fp_pipe without current_frame
        fp_pipe = {
            "frames": frames_dict
        }

        print(f"Output: frame index {current_frame} (out of {total_files} files)")
        if number_of_frames > 0:
            print(f"fp_pipe contains {len(frames_dict)} frames (indexes {skip_first_frames} to {end_frame - 1})")
        else:
            print(f"fp_pipe contains all {len(frames_dict)} frames")

        return selected_frame, fp_pipe