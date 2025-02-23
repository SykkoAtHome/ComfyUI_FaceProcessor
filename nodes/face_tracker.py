import cv2

from ..core.face_detector import FaceDetector
from ..core.image_processor import ImageProcessor


class FaceTracker:
    """ComfyUI node for tracking facial features across image sequences."""

    # Optical flow parameters optimized for face tracking
    OPTICAL_FLOW_PARAMS = {
        "pyramid_scale": 0.5,  # Each pyramid level is half the size of the previous
        "levels": 5,  # Number of pyramid levels
        "window_size": 21,  # Size of the search window at each pyramid level
        "iterations": 3,  # Number of iterations the algorithm does at each pyramid level
        "poly_n": 7,  # Size of the pixel neighborhood used for polynomial expansion
        "poly_sigma": 1.5,  # Standard deviation of the Gaussian used to smooth derivatives
        "flags": 0  # Optional flags (can be cv2.OPTFLOW_USE_INITIAL_FLOW or cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    }

    def __init__(self):
        self.face_detector = FaceDetector()
        self.image_processor = ImageProcessor()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fp_pipe": ("DICT",),
                "debug": ("BOOLEAN", {
                    "default": False
                }),
                "use_motion_vectors": ("BOOLEAN", {
                    "default": False
                }),
                "reset_points_interval": ("INT", {
                    "default": 10,
                    "min": 3,
                    "max": 50,
                    "step": 1
                }),
                "tracker_region_size": (["32", "48", "64", "96", "128"], {
                    "default": "64"
                }),
                "proxy_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.2,
                    "max": 1.0,
                    "step": 0.05
                }),
                "show_detection": ("BOOLEAN", {
                    "default": False
                }),
                "show_region": ("BOOLEAN", {
                    "default": False
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT")
    RETURN_NAMES = ("image", "fp_pipe")
    FUNCTION = "track_face"
    CATEGORY = "Face Processor"

    def track_face(self, image, fp_pipe, debug, use_motion_vectors, reset_points_interval,
                   tracker_region_size, proxy_scale, show_detection, show_region):
        input_image_np = self.image_processor.convert_to_numpy(image)
        try:
            # Convert input image for processing
            image_np = input_image_np
            if image_np is None:
                print("Error: Failed to convert input image")
                return image, fp_pipe

            # Apply proxy scale if needed
            if proxy_scale < 1.0:
                h, w = image_np.shape[:2]
                new_h, new_w = int(h * proxy_scale), int(w * proxy_scale)
                image_np = cv2.resize(image_np, (new_w, new_h))

            # Detect facial landmarks
            # TODO: Handle No face detected. Fill with NaN values and keep going
            landmarks_df = self.face_detector.detect_landmarks_mp(image_np)
            if landmarks_df is None:
                print("No face detected")
                return image, fp_pipe

            # If using proxy scale, rescale landmarks back to original size
            if proxy_scale < 1.0:
                landmarks_df['x'] = landmarks_df['x'] / proxy_scale
                landmarks_df['y'] = landmarks_df['y'] / proxy_scale

            # Update tracking data in fp_pipe
            current_frame = fp_pipe.get("current_frame", 0)
            frame_key = f"frame_{current_frame}"

            # Initialize tracking data structure if needed
            if "tracking_data" not in fp_pipe:
                fp_pipe["tracking_data"] = {}

            if frame_key not in fp_pipe["tracking_data"]:
                fp_pipe["tracking_data"][frame_key] = {}

            # Store MediaPipe detection results
            fp_pipe["tracking_data"][frame_key]["mediapipe"] = {
                "landmarks": landmarks_df.to_dict('records'),
                "timestamp": current_frame,
                "confidence": 1.0  # MediaPipe detection confidence
            }

            # Debug visualization
            if debug and show_detection:
                output_image = self._draw_debug_visualization(input_image_np, landmarks_df, show_region,
                                                              int(tracker_region_size))
                return self.image_processor.numpy_to_tensor(output_image), fp_pipe

            return image, fp_pipe

        except Exception as e:
            print(f"Error in track_face: {str(e)}")
            import traceback
            traceback.print_exc()
            return image, fp_pipe

    def _draw_debug_visualization(self, image, landmarks_df, show_region, region_size):
        """Draw debug visualization including landmarks and optionally tracking regions."""
        debug_image = image.copy()

        # Draw landmarks
        for _, landmark in landmarks_df.iterrows():
            x, y = int(landmark['x']), int(landmark['y'])
            cv2.circle(debug_image, (x, y), 2, (0, 255, 0), -1)

            # Draw tracking region if requested
            if show_region:
                half_size = region_size // 2
                x1 = max(0, x - half_size)
                y1 = max(0, y - half_size)
                x2 = min(image.shape[1], x + half_size)
                y2 = min(image.shape[0], y + half_size)
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

        return debug_image