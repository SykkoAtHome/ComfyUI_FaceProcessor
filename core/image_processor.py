import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from typing import Union, Optional, Tuple

class ImageProcessor:

    @staticmethod
    def calculate_face_bbox(landmarks_df: pd.DataFrame, padding_percent: float = 0.0) -> Optional[Tuple[int, int, int, int]]:
        """Calculate bounding box for face based on landmarks."""
        if landmarks_df is None or landmarks_df.empty:
            return None

        min_x = landmarks_df['x'].min()
        max_x = landmarks_df['x'].max()
        min_y = landmarks_df['y'].min()
        max_y = landmarks_df['y'].max()

        width = max_x - min_x
        height = max_y - min_y

        pad_x = width * padding_percent
        pad_y = height * padding_percent

        x1 = max(0, min_x - pad_x)
        y1 = max(0, min_y - pad_y)
        x2 = max_x + pad_x
        y2 = max_y + pad_y

        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    @staticmethod
    def resize_image(image: Union[torch.Tensor, np.ndarray, Image.Image], target_size: int) -> Optional[np.ndarray]:
        """Resize image to target size while maintaining aspect ratio and cropping to square."""
        if image is None:
            return None

        image_np = ImageProcessor.convert_to_numpy(image)
        h, w = image_np.shape[:2]

        # Determine the smaller dimension (height or width)
        min_dim = min(h, w)

        # Calculate the crop box to make the image square
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        end_x = start_x + min_dim
        end_y = start_y + min_dim

        # Crop the image to a square
        cropped_image = image_np[start_y:end_y, start_x:end_x]

        # Resize the cropped image to the target size
        resized = cv2.resize(cropped_image, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

        return resized

    @staticmethod
    def calculate_rotation_angle(landmarks_df: pd.DataFrame) -> float:
        """Calculate rotation angle based on eyes position."""
        if landmarks_df is None or landmarks_df.empty:
            return 0.0

        LEFT_EYE = 33  # Center of the left eye
        RIGHT_EYE = 263  # Center of the right eye

        left_eye = landmarks_df[landmarks_df['index'] == LEFT_EYE].iloc[0]
        right_eye = landmarks_df[landmarks_df['index'] == RIGHT_EYE].iloc[0]

        dx = right_eye['x'] - left_eye['x']
        dy = right_eye['y'] - left_eye['y']
        return np.degrees(np.arctan2(dy, dx))

    @staticmethod
    def rotate_image(image: Union[torch.Tensor, np.ndarray, Image.Image], landmarks_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
        """Rotate image based on facial landmarks."""
        image_np = ImageProcessor.convert_to_numpy(image)

        if image_np is None or landmarks_df is None:
            return None, None

        angle = ImageProcessor.calculate_rotation_angle(landmarks_df)
        height, width = image_np.shape[:2]
        center = (width // 2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image_np, rotation_matrix, (width, height), flags=cv2.INTER_LANCZOS4)

        # Transform landmarks
        ones = np.ones(shape=(len(landmarks_df), 1))
        points = np.hstack([landmarks_df[['x', 'y']].values, ones])
        transformed_points = rotation_matrix.dot(points.T).T

        updated_landmarks = landmarks_df.copy()
        updated_landmarks['x'] = transformed_points[:, 0]
        updated_landmarks['y'] = transformed_points[:, 1]

        return rotated_image, updated_landmarks

    @staticmethod
    def crop_face_to_square(image: np.ndarray, landmarks_df: pd.DataFrame, padding_percent: float = 0.0) -> Tuple[
        Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Crop face region to a square (1:1) based on landmarks."""
        bbox = ImageProcessor.calculate_face_bbox(landmarks_df, padding_percent)
        if bbox is None:
            return None, None

        x, y, w, h = bbox

        # Calculate the center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2

        # Determine the size of the square crop
        crop_size = max(w, h)
        half_size = crop_size // 2

        # Calculate the crop coordinates
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(image.shape[1], center_x + half_size)
        y2 = min(image.shape[0], center_y + half_size)

        # Adjust if the crop goes out of bounds
        if x2 - x1 < crop_size:
            x1 = max(0, x2 - crop_size)
        if y2 - y1 < crop_size:
            y1 = max(0, y2 - crop_size)

        # Crop the image
        cropped_face = image[y1:y2, x1:x2]

        # Return the cropped face and the crop bounding box
        return cropped_face, (x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def draw_landmarks(image_size: Union[int, Tuple[int, int]],
                       landmarks_df: pd.DataFrame,
                       transparency: float = 0.5,
                       color: Tuple[int, int, int] = (0, 255, 0),
                       radius: int = 2,
                       label: bool = False) -> Union[np.ndarray, None]:
        """
        Draw facial landmarks on a transparent image.

        Args:
            image_size: Target image size (int or tuple of width, height)
            landmarks_df: DataFrame containing landmark coordinates
            transparency: Alpha channel value (0.0-1.0)
            color: RGB color tuple for landmarks
            radius: Radius of landmark points
            label: Whether to show landmark IDs

        Returns:
            numpy array: RGBA image with drawn landmarks
        """
        if landmarks_df is None or landmarks_df.empty:
            return None

        # Create transparent image
        if isinstance(image_size, tuple):
            width, height = image_size
        else:
            width = height = image_size

        # Create RGBA image (alpha channel for transparency)
        image = np.zeros((height, width, 4), dtype=np.uint8)

        # Set alpha channel based on transparency
        alpha = int(transparency * 255)

        # Draw landmarks
        for _, landmark in landmarks_df.iterrows():
            x = int(landmark['x'])
            y = int(landmark['y'])
            index = int(landmark['index'])

            # Ensure coordinates are within image bounds
            if 0 <= x < width and 0 <= y < height:
                # Draw filled circle
                cv2.circle(
                    image,
                    (x, y),
                    radius,
                    (*color, alpha),
                    -1,  # Filled circle
                    cv2.LINE_AA
                )

                # Add landmark ID if requested
                if label:
                    # Text parameters
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.4
                    text_thickness = 1

                    # Add small offset to text position
                    text_x = x + radius + 5
                    text_y = y + radius

                    # Draw text with white color and black outline for better visibility
                    text = str(index)

                    # Draw text outline
                    cv2.putText(
                        image,
                        text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        (0, 0, 0, alpha),  # Black outline
                        text_thickness + 1,
                        cv2.LINE_AA
                    )

                    # Draw text
                    cv2.putText(
                        image,
                        text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        (*color, alpha),  # Main color
                        text_thickness,
                        cv2.LINE_AA
                    )

        return image

    @staticmethod
    def draw_dynamic_histogram(composite: np.ndarray,
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

    # Image Converters
    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array.

        Args:
            tensor: Input tensor in format (B,H,W,C) or (B,C,H,W), values 0-1

        Returns:
            np.ndarray: RGB image (H,W,3), uint8 values 0-255
        """
        if tensor is None:
            return None

        # Handle different tensor formats
        img = tensor.detach().cpu().numpy()
        if len(img.shape) == 4:
            img = img[0]  # Remove batch dimension

        # Handle channel dimension if in (C,H,W) format
        if img.shape[0] in [1, 3, 4]:
            img = img.transpose(1, 2, 0)

        # Handle single channel
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = np.stack((img,) * 3, axis=-1)

        # Handle RGBA
        elif img.shape[2] == 4:
            img = img[..., :3]

        # Convert to uint8 if float
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).clip(0, 255).astype(np.uint8)

        return img

    @staticmethod
    def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor.

        Args:
            array: RGB image (H,W,3), uint8 or float32

        Returns:
            torch.Tensor: (1,H,W,3) tensor, values 0-1
        """
        if array is None:
            return None

        # Convert to float32 if uint8
        if array.dtype == np.uint8:
            array = array.astype(np.float32) / 255.0

        # Add batch dimension
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor

    @staticmethod
    def pil_to_numpy(image: Image.Image) -> np.ndarray:
        """Convert PIL image to numpy array.

        Args:
            image: PIL Image

        Returns:
            np.ndarray: RGB image (H,W,3), uint8 values 0-255
        """
        if image is None:
            return None

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return np.array(image)

    @staticmethod
    def numpy_to_pil(array: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL image.

        Args:
            array: RGB image (H,W,3), uint8 or float32

        Returns:
            PIL.Image: RGB PIL image
        """
        if array is None:
            return None

        # Convert to uint8 if float
        if array.dtype == np.float32 or array.dtype == np.float64:
            array = (array * 255).clip(0, 255).astype(np.uint8)

        return Image.fromarray(array)

    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert PyTorch tensor to PIL image.

        Args:
            tensor: Input tensor in format (B,H,W,C), values 0-1

        Returns:
            PIL.Image: RGB PIL image
        """
        return ImageProcessor.numpy_to_pil(ImageProcessor.tensor_to_numpy(tensor))

    @staticmethod
    def pil_to_tensor(image: Image.Image) -> torch.Tensor:
        """Convert PIL image to PyTorch tensor.

        Args:
            image: PIL Image

        Returns:
            torch.Tensor: (1,H,W,3) tensor, values 0-1
        """
        return ImageProcessor.numpy_to_tensor(ImageProcessor.pil_to_numpy(image))

    @staticmethod
    def convert_to_numpy(image: Union[torch.Tensor, np.ndarray, Image.Image]) -> np.ndarray:
        """Universal converter to numpy array.

        Args:
            image: Input in any supported format

        Returns:
            np.ndarray: RGB image (H,W,3), uint8 values 0-255
        """
        if image is None:
            return None

        if torch.is_tensor(image):
            return ImageProcessor.tensor_to_numpy(image)
        elif isinstance(image, Image.Image):
            return ImageProcessor.pil_to_numpy(image)
        elif isinstance(image, np.ndarray):
            # Handle grayscale
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # Handle RGBA
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            # Convert to uint8 if float
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    # Mask conversion methods
    @staticmethod
    def convert_mask_to_numpy(mask: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert mask to numpy array.

        Args:
            mask: Input mask as tensor (1,H,W) or array

        Returns:
            np.ndarray: Mask as (H,W) uint8 array, values 0-255
        """
        if mask is None:
            return None

        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
            if len(mask.shape) == 3:
                mask = mask[0]  # Remove batch dimension

        if mask.dtype == np.float32 or mask.dtype == np.float64:
            mask = (mask * 255).clip(0, 255).astype(np.uint8)

        return mask

    @staticmethod
    def convert_mask_to_tensor(mask: np.ndarray) -> Tensor | None:
        """Convert numpy mask to tensor.

        Args:
            mask: Input mask as (H,W) array

        Returns:
            torch.Tensor: Mask as (1,H,W) tensor, values 0-1
        """
        if mask is None:
            return None

        if mask.dtype == np.uint8:
            mask = mask.astype(np.float32) / 255.0

        return torch.from_numpy(mask).unsqueeze(0)

    @staticmethod
    def generate_face_mask(image_size: tuple, landmarks_df: pd.DataFrame) -> np.ndarray:
        """
        Generate a binary mask covering all facial landmarks using the convex hull algorithm.

        Args:
            image_size: Tuple (width, height) of the target mask.
            landmarks_df: DataFrame containing landmark coordinates with columns 'x', 'y'.

        Returns:
            numpy.ndarray: Binary mask as a single-channel array (values 0 or 1, dtype float32).
        """
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            raise ValueError("image_size must be a tuple (width, height).")

        if not isinstance(landmarks_df, pd.DataFrame) or landmarks_df.empty:
            print("No valid landmarks provided for mask generation.")
            return np.zeros((image_size[1], image_size[0]), dtype=np.float32)

        try:
            width, height = image_size
            mask = np.zeros((height, width), dtype=np.uint8)

            # Extract valid points and ensure they are within bounds
            points = landmarks_df[['x', 'y']].to_numpy(dtype=np.int32)
            valid_mask = (0 <= points[:, 0]) & (points[:, 0] < width) & (0 <= points[:, 1]) & (points[:, 1] < height)
            valid_points = points[valid_mask]

            if valid_points.shape[0] < 3:
                print(f"Not enough valid landmarks for mask generation (found {valid_points.shape[0]}).")
                return mask.astype(np.float32)

            # Generate convex hull and fill the mask
            hull = cv2.convexHull(valid_points)
            cv2.fillConvexPoly(mask, hull, color=1)

            return mask.astype(np.float32)

        except Exception as e:
            print(f"Error generating face mask: {e}")
            return np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    @staticmethod
    def mask_interpolation(prev_mask: np.ndarray, next_mask: np.ndarray) -> np.ndarray:
        """
        Interpolate between two masks to create a new mask.

        Args:
            prev_mask: Previous mask as numpy array (values 0-1)
            next_mask: Next mask as numpy array (values 0-1)

        Returns:
            numpy.ndarray: Interpolated mask with values normalized to 0 or 1
        """
        if prev_mask is None or next_mask is None:
            raise ValueError("Both masks must be provided for interpolation")

        if prev_mask.shape != next_mask.shape:
            raise ValueError(f"Mask shapes don't match: {prev_mask.shape} vs {next_mask.shape}")

        # Create average mask by interpolating values
        interpolated = (prev_mask + next_mask) / 2

        # Normalize to binary mask (any value > 0 becomes 1)
        binary_mask = (interpolated > 0).astype(np.float32)

        return binary_mask

    # Loaders
    @staticmethod
    def load_image_from_path(image_path):
        """Load and convert image from file path to tensor format."""
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return ImageProcessor.pil_to_tensor(image)
        except Exception as e:
            print(f"Error loading image from {image_path}: {str(e)}")
            return None

    def calculate_blur_score(self, image, method='gradient', normalize=True, invert_scale=True):
        """
        Calculate the blur score of an image using various methods.

        Args:
            image: Input image in various formats (torch.Tensor, numpy.ndarray, PIL.Image)
            method: Blur detection method ('gradient' or 'fft')
            normalize: Whether to normalize score to 0-1 range
            invert_scale: If True, higher values mean more blur. If False, higher values mean less blur (sharper)

        Returns:
            tuple: (blur_score, normalized_score)
                - blur_score: Raw blur metric value
                - normalized_score: Normalized value between 0-1 with 4 decimal places
        """
        # Convert input to numpy array
        image_np = self.convert_to_numpy(image)
        if image_np is None:
            print("Error: Failed to convert image for blur calculation")
            return 0.0, 0.0

        # Convert to grayscale if needed
        if len(image_np.shape) == 3 and image_np.shape[2] >= 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        # Calculate raw blur score based on selected method
        if method == 'gradient':
            score = self._calculate_blur_gradient(gray)
        elif method == 'fft':
            score = self._calculate_blur_fft(gray)
        else:
            print(f"Error: Unknown blur detection method '{method}'")
            return 0.0, 0.0

        # If batch processing is needed, we should dynamically find min/max
        # from the entire batch. Since we're processing one image at a time here,
        # we'll use previously observed ranges with 10% margin.

        # Get current min/max values from class attributes or use default
        if hasattr(self, '_blur_min_max') and method in self._blur_min_max:
            min_val, max_val = self._blur_min_max[method]
        else:
            # Initialize if not exists
            if not hasattr(self, '_blur_min_max'):
                self._blur_min_max = {}

            # Default values if we don't have historical data
            if method == 'gradient':
                # Starting values based on observations
                min_val, max_val = 6.0, 16.0
            else:  # fft
                # Starting values based on observations
                min_val, max_val = 0.003, 0.015

            # Store initial values
            self._blur_min_max[method] = (min_val, max_val)

        # Update min/max with current score (to adapt over time)
        current_min, current_max = self._blur_min_max[method]
        if score < current_min:
            current_min = score
        if score > current_max:
            current_max = score

        # Store updated values
        self._blur_min_max[method] = (current_min, current_max)

        # Apply 10% margin to min/max for normalization
        min_with_margin = current_min * 0.9
        max_with_margin = current_max * 1.1

        # Normalize the score to 0-1 range
        if max_with_margin - min_with_margin < 1e-10:  # Avoid division by zero
            normalized = 0.5  # Default to middle if range is too small
        else:
            normalized = (score - min_with_margin) / (max_with_margin - min_with_margin)
            normalized = min(1.0, max(0.0, normalized))  # Clamp to 0-1

        # Invert scale if requested (higher value = more blur)
        if invert_scale:
            normalized = 1.0 - normalized

        # Round to 4 decimal places
        normalized = round(normalized, 4)

        # Return raw score and normalized value
        return score, normalized

    def _calculate_blur_gradient(self, gray):
        """
        Calculate blur amount using gradient method.

        Args:
            gray: Grayscale input image (numpy array)

        Returns:
            float: Blur metric value (higher = sharper)
        """
        # Calculate gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate gradient magnitude
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # Return mean gradient magnitude
        return np.mean(magnitude)

    def _calculate_blur_fft(self, gray):
        """
        Calculate blur amount using FFT method.

        Args:
            gray: Grayscale input image (numpy array)

        Returns:
            float: Blur metric value (higher = sharper)
        """
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        # Apply FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

        # Create a high-pass filter (keep frequencies above 80% of nyquist)
        mask = np.zeros((rows, cols), np.uint8)
        r = min(rows, cols) // 5  # radius of the circle
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 1

        # Apply mask and compute magnitude spectrum
        fshift_filtered = fshift * (1 - mask)
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # Normalize and return score
        # Higher value means more high frequencies (sharper image)
        return np.mean(img_back) / np.mean(gray)


