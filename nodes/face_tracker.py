import cv2
import numpy as np
import os
import traceback

from ..core.face_detector import FaceDetector
from ..core.image_processor import ImageProcessor


class FaceTracker:
    """ComfyUI node for tracking facial features across image sequences."""

    def __init__(self):
        self.face_detector = FaceDetector()
        self.image_processor = ImageProcessor()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fp_pipe": ("DICT",),
                "use_optical_flow": ("BOOLEAN", {
                    "default": False
                }),
                "use_track_points": ("BOOLEAN", {
                    "default": False
                }),
                "proxy_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.2,
                    "max": 1.0,
                    "step": 0.05
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT")
    RETURN_NAMES = ("image", "fp_pipe")
    FUNCTION = "track_face"
    CATEGORY = "Face Processor"

    def track_face(self, image, fp_pipe, use_optical_flow, use_track_points, proxy_scale):
        """
        Main tracking method processing all frames in the batch.
        """
        try:
            # Process all frames in the batch
            batch_size = image.shape[0] if len(image.shape) == 4 else 1

            for frame_idx in range(batch_size):
                # Get current frame
                current_frame = image[frame_idx:frame_idx + 1] if batch_size > 1 else image
                frame_np = self.image_processor.convert_to_numpy(current_frame)

                if frame_np is None:
                    print(f"Error: Failed to convert frame {frame_idx}")
                    continue

                self._process_single_frame(
                    frame_np,
                    frame_idx,
                    fp_pipe,
                    use_optical_flow,
                    use_track_points,
                    proxy_scale
                )

            # Return original image and updated fp_pipe
            return image, fp_pipe

        except Exception as e:
            print(f"Error in track_face: {str(e)}")
            traceback.print_exc()
            return image, fp_pipe

    def _process_single_frame(self, frame_np, frame_idx, fp_pipe, use_optical_flow, use_track_points, proxy_scale):
        """
        Process a single frame applying selected tracking methods based on user preferences.
        MediaPipe detection is always performed as it's obligatory.
        """
        frame_key = f"frame_{frame_idx}"

        # Ensure frame exists in fp_pipe
        if frame_key not in fp_pipe["frames"]:
            fp_pipe["frames"][frame_key] = {}

        # Initialize tracking data if needed
        if "tracking" not in fp_pipe["frames"][frame_key]:
            fp_pipe["frames"][frame_key]["tracking"] = {}

        # Step 1: MediaPipe Detection (always performed)
        print(f"Frame {frame_idx}: Running MediaPipe detection")
        mediapipe_results = self._mediapipe_detector(frame_np, proxy_scale)
        if mediapipe_results:
            fp_pipe["frames"][frame_key]["tracking"]["mediapipe"] = mediapipe_results
        else:
            print(f"Frame {frame_idx}: MediaPipe detection failed")
            return

        # Step 2: Optical Flow (if enabled)
        if use_optical_flow and frame_idx > 0:
            print(f"Frame {frame_idx}: Adding Optical Flow placeholder")
            optical_flow_results = self._optical_flow_tracker(mediapipe_results)
            if optical_flow_results:
                fp_pipe["frames"][frame_key]["tracking"]["optical_flow"] = optical_flow_results

        # Step 3: Points Tracking (if enabled)
        if use_track_points:
            print(f"Frame {frame_idx}: Adding Points Tracking placeholder")
            points_tracking_results = self._points_tracker(mediapipe_results)
            if points_tracking_results:
                fp_pipe["frames"][frame_key]["tracking"]["points_tracker"] = points_tracking_results

    def _mediapipe_detector(self, image_np, proxy_scale):
        """
        Detect facial landmarks using MediaPipe.
        Returns detection results in unified format where key is landmark ID and value is
        a dictionary with x, y coordinates and confidence.
        """
        try:
            # Apply proxy scale if needed
            working_image = self._apply_proxy_scale(image_np, proxy_scale)

            landmarks_df = self.face_detector.detect_landmarks_mp(working_image)
            if landmarks_df is None:
                return None

            # Rescale landmarks if proxy scale was used
            if proxy_scale < 1.0:
                landmarks_df['x'] = landmarks_df['x'] / proxy_scale
                landmarks_df['y'] = landmarks_df['y'] / proxy_scale

            # Convert to unified format
            results = {}
            for _, row in landmarks_df.iterrows():
                landmark_id = int(row['index'])
                results[landmark_id] = {
                    "x": float(row['x']),
                    "y": float(row['y'])
                }

            return results

        except Exception as e:
            print(f"Error in MediaPipe detection: {str(e)}")
            return None

    def _optical_flow_tracker(self, mediapipe_results):
        """
        Placeholder for optical flow functionality.
        Will be implemented in the future.
        """
        if not mediapipe_results:
            return None

        # Create placeholder data based on MediaPipe results
        results = {}

        # For each landmark, create placeholder optical flow data
        for landmark_id, data in mediapipe_results.items():
            results[landmark_id] = {
                "x": data["x"],
                "y": data["y"],
                "dx": 0.0,  # Placeholder flow in x direction
                "dy": 0.0  # Placeholder flow in y direction
            }

        return results

    def _points_tracker(self, mediapipe_results):
        """
        Placeholder for points tracker functionality.
        Will be implemented in the future.
        """
        if not mediapipe_results:
            return None

        # Create placeholder data based on MediaPipe results
        results = {}

        # For each landmark, create placeholder points tracker data
        for landmark_id, data in mediapipe_results.items():
            results[landmark_id] = {
                "x": data["x"],
                "y": data["y"]
            }

        # Add reset flag (always False in placeholder)
        results["reset_occurred"] = False

        return results

    def _apply_proxy_scale(self, image_np, proxy_scale):
        """Helper method to apply proxy scale to image."""
        if proxy_scale < 1.0:
            h, w = image_np.shape[:2]
            new_h, new_w = int(h * proxy_scale), int(w * proxy_scale)
            return cv2.resize(image_np, (new_w, new_h))
        return image_np