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
                "frame_number": ("INT", {
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
    RETURN_NAMES = ("image", "image_sequence")
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

    def feed_images(self, directory: str, frame_number: int, number_of_frames: int = 0, skip_first_frames: int = 0) -> \
    Tuple[torch.Tensor, Dict]:
        """
        Main processing function

        Args:
            directory: Path to images directory
            frame_number: Index of frame to return
            number_of_frames: Number of frames to process (0 = all frames)
            skip_first_frames: Number of frames to skip from the beginning

        Returns:
            Tuple containing:
            - Selected frame as tensor
            - Dictionary with image_sequence
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
            return empty_image, {"frames": {}, "current_frame": 0}

        # Calculate frame ranges
        total_available_frames = len(self.image_files)
        start_frame = skip_first_frames

        # If number_of_frames is 0, use all remaining frames
        if number_of_frames == 0:
            end_frame = total_available_frames
        else:
            end_frame = min(start_frame + number_of_frames, total_available_frames)

        # Prepare frame mapping with sequential indices starting from 0
        frames_dict = {}
        file_mapping = {}  # Store original indices for debug purposes

        for new_idx, original_idx in enumerate(range(start_frame, end_frame)):
            frames_dict[new_idx] = self.image_files[original_idx]
            file_mapping[new_idx] = original_idx  # For debugging

        # Print debug info
        print(f"Processing frames {start_frame} to {end_frame - 1}")
        print(f"Remapped to indices 0 to {len(frames_dict) - 1}")

        # Prepare return data
        image_sequence = {
            "frames": frames_dict,
            "current_frame": min(frame_number, len(frames_dict) - 1),
            "original_indices": file_mapping  # Optional, for debugging
        }

        # Validate frame number
        if frame_number >= len(frames_dict):
            frame_number = len(frames_dict) - 1
            print(f"Frame number too high, using last available frame: {frame_number}")
        elif frame_number < 0:
            frame_number = 0
            print(f"Frame number too low, using first available frame: {frame_number}")

        # Update current frame in sequence data
        image_sequence["current_frame"] = frame_number

        # Load selected frame
        selected_frame = self._load_image(frames_dict[frame_number])

        if selected_frame is None:
            print(f"Failed to load frame {frame_number}")
            empty_image = torch.zeros((1, 64, 64, 3))
            return empty_image, image_sequence

        return selected_frame, image_sequence
