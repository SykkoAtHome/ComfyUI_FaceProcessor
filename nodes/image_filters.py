import torch
import numpy as np
import cv2

from torch import Tensor
from ..core.image_processor import ImageProcessor

class HighPassFilter:
    """ComfyUI node implementing AE-style high-pass filter with dynamic histogram visualization"""

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
                }),
                "show_histogram": ("BOOLEAN", {
                    "default": False
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT")
    RETURN_NAMES = ("image", "HPF Settings")
    FUNCTION = "apply_hpf"
    CATEGORY = "Face Processor/Tools"

    def apply_hpf(self, image: torch.Tensor, blur_radius: int, blur_iterations: int,
                  blend_opacity: float, input_black: int, input_white: int,
                  gamma: float, show_histogram: bool) -> tuple[Tensor, dict]:
        """Main processing function implementing the High Pass Filter"""
        # Convert tensor to numpy array
        np_img = image[0].numpy() * 255
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)

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

        # Add histogram visualization if enabled
        if show_histogram:
            hist_img = ImageProcessor.draw_dynamic_histogram(composite, in_black, in_white, gamma)
            target_width = result.shape[1]
            hist_img = cv2.resize(hist_img, (target_width, 200))
            result = np.vstack([result, hist_img])

        # Convert back to tensor format
        result = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)

        # Prepare settings dictionary
        settings = {
            "high_pass_filter": {
                "blur_radius": blur_radius,
                "blur_iterations": blur_iterations,
                "blend_opacity": blend_opacity,
                "input_black": input_black,
                "input_white": input_white,
                "gamma": gamma,
                "show_histogram": False
            }
        }

        return result, settings
