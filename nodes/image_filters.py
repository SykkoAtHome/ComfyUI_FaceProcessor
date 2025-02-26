import torch
import numpy as np
import cv2
from torch import Tensor
from ..core.image_processor import ImageProcessor


class HighPassFilter:
    """ComfyUI node implementing AE-style high-pass filter"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 20,
                    "step": 1
                }),
                "blur_iterations": ("INT", {
                    "default": 6,
                    "min": 1,
                    "max": 20,
                    "step": 1
                }),
                "blend_opacity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.45,
                    "max": 0.55,
                    "step": 0.01
                }),
                "input_black": ("INT", {
                    "default": 117,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "input_white": ("INT", {
                    "default": 137,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT")
    RETURN_NAMES = ("image", "HPF Settings")
    FUNCTION = "apply_hpf"
    CATEGORY = "Face Processor/Tools"

    def apply_hpf(self, image: torch.Tensor, blur_radius: int, blur_iterations: int,
                  blend_opacity: float, input_black: int, input_white: int,
                  gamma: float) -> tuple[Tensor, dict]:
        """Main processing function implementing the High Pass Filter

        Args:
            image: Input image tensor (B,H,W,C) or (H,W,C)
            blur_radius: Radius for box blur (controls blur kernel size)
            blur_iterations: Number of blur iterations (controls blur strength)
            blend_opacity: Opacity for blending original with inverted blurred image
            input_black: Black level for tone mapping (0-255)
            input_white: White level for tone mapping (0-255)
            gamma: Gamma correction value

        Returns:
            tuple[Tensor, dict]: Processed image and settings dictionary
        """
        # Check if input is a batch of images
        is_batch = len(image.shape) == 4
        batch_size = image.shape[0] if is_batch else 1

        # Store results
        results = []

        # Process each image in the batch (or the single image)
        for i in range(batch_size):
            # Get current image
            current_image = image[i:i + 1] if is_batch else image

            # Convert input tensor to numpy array
            np_img = ImageProcessor.tensor_to_numpy(current_image)

            # 1. Apply box blur with multiple iterations
            kernel_size = 2 * blur_radius + 1
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
            blurred = np_img.copy()
            for _ in range(blur_iterations):
                blurred = cv2.filter2D(blurred, -1, kernel, borderType=cv2.BORDER_REPLICATE)

            # 2. Invert blurred image
            inverted = cv2.bitwise_not(blurred)

            # 3. Blend original with inverted using specified opacity
            composite = cv2.addWeighted(np_img, 1 - blend_opacity, inverted, blend_opacity, 0)

            # 4. Apply levels adjustment
            in_black = np.clip(input_black, 0, 255)
            in_white = np.clip(input_white, in_black + 1, 255)
            levels = np.clip((composite.astype(np.float32) - in_black) / (in_white - in_black), 0, 1)
            levels = np.power(levels, 1.0 / gamma) if gamma > 0 else levels
            result = (levels * 255).astype(np.uint8)

            # Convert back to tensor format
            result_tensor = ImageProcessor.numpy_to_tensor(result)
            results.append(result_tensor)

        # Combine results
        if is_batch:
            final_result = torch.cat(results, dim=0)
        else:
            final_result = results[0]

        # Prepare settings dictionary
        settings = {
            "high_pass_filter": {
                "blur_radius": blur_radius,
                "blur_iterations": blur_iterations,
                "blend_opacity": blend_opacity,
                "input_black": input_black,
                "input_white": input_white,
                "gamma": gamma
            }
        }

        return final_result, settings
