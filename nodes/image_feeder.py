import os
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Union, Tuple


class ImageFeeder:
    """ComfyUI node for feeding images from a directory."""

    def __init__(self):
        self.current_dir = None
        self.image_files = []

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
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT")
    RETURN_NAMES = ("image", "frames_data")
    FUNCTION = "feed_images"
    CATEGORY = "image"

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

            # Convert to numpy array and normalize
            image_np = np.array(image).astype(np.float32) / 255.0

            # Convert to torch tensor and add batch dimension
            return torch.from_numpy(image_np).unsqueeze(0)

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

    def feed_images(self, directory: str, frame_number: int) -> Tuple[torch.Tensor, Dict]:
        """
        Main processing function

        Args:
            directory: Path to images directory
            frame_number: Index of frame to return

        Returns:
            Tuple containing:
            - Selected frame as tensor
            - Dictionary with frames data
        """
        # Check if directory changed
        if directory != self.current_dir:
            print(f"New directory detected, scanning: {directory}")
            self.image_files = self._scan_directory(directory)
            self.current_dir = directory

        # Prepare return data with frame to file mapping
        frames_data = {
            "frames": {
                idx: file_path
                for idx, file_path in enumerate(self.image_files)
            },
            "current_frame": frame_number
        }

        # Handle empty directory case
        if not self.image_files:
            print("No images found in directory")
            # Return empty image and data
            empty_image = torch.zeros((1, 64, 64, 3))
            return (empty_image, frames_data)

        # Validate frame number
        max_frame = len(self.image_files) - 1
        if frame_number > max_frame:
            print(f"Warning: Requested frame {frame_number} is out of range. Using last available frame ({max_frame})")
            frame_number = max_frame
        elif frame_number < 0:
            print(f"Warning: Requested frame {frame_number} is negative. Using first frame (0)")
            frame_number = 0

        # Load selected frame
        selected_frame = self._load_image(self.image_files[frame_number])

        if selected_frame is None:
            print(f"Failed to load frame {frame_number}")
            empty_image = torch.zeros((1, 64, 64, 3))
            return (empty_image, frames_data)

        return (selected_frame, frames_data)
