import torch
import numpy as np
import cv2

from torch import Tensor


class HighPassFilter:
    """ComfyUI node implementing AE-style high-pass filter with dynamic histogram visualization"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "blur_iterations": ("INT", {"default": 6, "min": 1, "max": 20, "step": 1}),
                "blend_opacity": ("FLOAT", {"default": 0.5, "min": 0.45, "max": 0.55, "step": 0.01}),
                "input_black": ("INT", {"default": 117, "min": 0, "max": 255, "step": 1}),
                "input_white": ("INT", {"default": 137, "min": 0, "max": 255, "step": 1}),
                "gamma": ("FLOAT", {"default": 0, "min": 0.5, "max": 2, "step": 0.01}),
                "show_histogram": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_effect"
    CATEGORY = "Face Processor/Tools"

    def draw_dynamic_histogram(self, composite: np.ndarray,
                               input_black: int, input_white: int,
                               gamma: float) -> np.ndarray:
        """Creates histogram visualization with dynamic control lines"""
        # Convert to grayscale for histogram calculation
        gray = cv2.cvtColor(composite, cv2.COLOR_RGB2GRAY) if len(composite.shape) == 3 else composite

        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_img = np.zeros((200, 256, 3), dtype=np.uint8)
        cv2.normalize(hist, hist, 0, hist_img.shape[0], cv2.NORM_MINMAX)

        # Draw histogram baseline
        for i in range(1, 256):
            cv2.line(hist_img,
                     (i - 1, 200 - int(hist[i - 1])),
                     (i, 200 - int(hist[i])),
                     (128, 128, 128), 1)

        # Calculate gamma position
        gamma_pos = int(255 * (0.5 ** (1 / gamma)))

        # Draw control lines
        cv2.line(hist_img, (input_black, 0), (input_black, 200), (255, 0, 0), 2)  # Blue - black level
        cv2.line(hist_img, (input_white, 0), (input_white, 200), (0, 255, 0), 2)  # Green - white level
        cv2.line(hist_img, (gamma_pos, 0), (gamma_pos, 200), (255, 255, 255), 2)  # White - gamma

        # Add legend
        cv2.putText(hist_img, f"Black: {input_black}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(hist_img, f"White: {input_white}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(hist_img, f"Gamma: {gamma:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return hist_img

    def apply_effect(self, image: torch.Tensor, blur_radius: int, blur_iterations: int,
                     blend_opacity: float, input_black: int, input_white: int,
                     gamma: float, show_histogram: bool) -> tuple[Tensor]:
        """Main processing function implementing the AE-style effect workflow"""
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
        levels = np.power(levels, 1.0 / gamma)
        result = (levels * 255).astype(np.uint8)

        # Add histogram visualization if enabled
        if show_histogram:
            hist_img = self.draw_dynamic_histogram(composite, in_black, in_white, gamma)
            target_width = result.shape[1]
            hist_img = cv2.resize(hist_img, (target_width, 200))
            result = np.vstack([result, hist_img])

        # Convert back to tensor format
        result = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)

        return (result,)