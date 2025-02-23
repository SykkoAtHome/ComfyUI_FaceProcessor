import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from typing import Union, Optional, Tuple

class ImageProcessor:
    @staticmethod
    def convert_to_numpy(image: Union[torch.Tensor, np.ndarray, Image.Image]) -> np.ndarray:
        """Convert different image types to numpy array."""
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
            if len(image.shape) == 4:
                image = image[0]
            image = (image * 255).astype(np.uint8)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            if image.dtype == np.float32 and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 4:
                image = image[0]

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        return image

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
