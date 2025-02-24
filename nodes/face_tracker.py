import cv2
import numpy as np
import os
import traceback
import json
from datetime import datetime

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
        self._prev_frame_data = {}

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
        """
        Main tracking method processing all frames in the batch.
        If debug=True, performs visualization of tracking results.
        """
        try:
            # Process all frames in the batch regardless of debug mode
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
                    use_motion_vectors,
                    reset_points_interval,
                    int(tracker_region_size),
                    proxy_scale
                )

            # Save fp_pipe state (regardless of debug mode)
            self._save_fp_pipe_state(fp_pipe)

            # Return visualization if in debug mode and show_detection is enabled
            if debug and show_detection:
                current_frame_idx = fp_pipe.get("current_frame", 0)
                frame_key = f"frame_{current_frame_idx}"

                if frame_key in fp_pipe["frames"] and "tracking" in fp_pipe["frames"][frame_key]:
                    # Create visualization only for the current frame
                    current_frame_np = self.image_processor.convert_to_numpy(
                        image[current_frame_idx:current_frame_idx + 1] if batch_size > 1 else image
                    )

                    output_image = self._draw_debug_visualization(
                        current_frame_np,
                        fp_pipe["frames"][frame_key]["tracking"],
                        show_region,
                        int(tracker_region_size)
                    )
                    return self.image_processor.numpy_to_tensor(output_image), fp_pipe

            # Return original image and updated fp_pipe if not in debug mode or if visualization failed
            return image, fp_pipe

        except Exception as e:
            print(f"Error in track_face: {str(e)}")
            traceback.print_exc()
            return image, fp_pipe

    def _process_single_frame(self, frame_np, frame_idx, fp_pipe, use_motion_vectors,
                              reset_points_interval, region_size, proxy_scale):
        """
        Process a single frame applying all tracking methods as configured.
        """
        frame_key = f"frame_{frame_idx}"

        # Ensure frame exists in fp_pipe
        if frame_key not in fp_pipe["frames"]:
            fp_pipe["frames"][frame_key] = {}

        # Initialize tracking data if needed
        if "tracking" not in fp_pipe["frames"][frame_key]:
            fp_pipe["frames"][frame_key]["tracking"] = {}

        # Step 1: MediaPipe Detection
        mediapipe_results = self._mediapipe_detector(frame_np, proxy_scale)
        if mediapipe_results:
            fp_pipe["frames"][frame_key]["tracking"]["mediapipe"] = mediapipe_results

        # Step 2: Optical Flow (if enabled)
        if use_motion_vectors and frame_idx > 0:
            prev_frame_key = f"frame_{frame_idx - 1}"
            # Get previous frame from fp_pipe
            if prev_frame_key in fp_pipe["frames"]:
                # Get previous frame path and load it
                prev_frame_path = fp_pipe["frames"][prev_frame_key].get("original_image_path")
                if prev_frame_path and os.path.exists(prev_frame_path):
                    prev_frame = cv2.imread(prev_frame_path)
                    if prev_frame is not None:
                        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)

                        # Get landmarks from previous frame's mediapipe tracking
                        if "mediapipe" in fp_pipe["frames"][prev_frame_key].get("tracking", {}):
                            prev_landmarks = fp_pipe["frames"][prev_frame_key]["tracking"]["mediapipe"]

                            # Get landmarks from current frame's mediapipe tracking
                            if "mediapipe" in fp_pipe["frames"][frame_key].get("tracking", {}):
                                curr_landmarks = fp_pipe["frames"][frame_key]["tracking"]["mediapipe"]

                                optical_flow_results = self._optical_flow_tracker(
                                    prev_frame,
                                    frame_np,
                                    prev_landmarks,
                                    curr_landmarks
                                )
                                if optical_flow_results:
                                    fp_pipe["frames"][frame_key]["tracking"]["optical_flow"] = optical_flow_results

        # Step 3: Points Tracking
        if "mediapipe" in fp_pipe["frames"][frame_key].get("tracking", {}):
            points_tracking_results = self._points_tracker(
                frame_np,
                fp_pipe["frames"][frame_key]["tracking"]["mediapipe"],
                region_size,
                frame_idx,
                reset_points_interval
            )
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
                    "y": float(row['y']),
                    "confidence": 1.0  # MediaPipe doesn't provide per-landmark confidence
                }

            return results

        except Exception as e:
            print(f"Error in MediaPipe detection: {str(e)}")
            return None

    def _optical_flow_tracker(self, prev_frame, curr_frame, prev_landmarks, curr_landmarks):
        """
        Calculate optical flow between consecutive frames.
        Returns flow vectors for tracked landmarks.
        """
        try:
            # Convert frames to grayscale for optical flow
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

            # Initialize flow matrix with zeros
            flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)

            # Calculate Farneback optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                flow,  # Pre-allocated flow matrix
                self.OPTICAL_FLOW_PARAMS["pyramid_scale"],
                self.OPTICAL_FLOW_PARAMS["levels"],
                self.OPTICAL_FLOW_PARAMS["window_size"],
                self.OPTICAL_FLOW_PARAMS["iterations"],
                self.OPTICAL_FLOW_PARAMS["poly_n"],
                self.OPTICAL_FLOW_PARAMS["poly_sigma"],
                self.OPTICAL_FLOW_PARAMS["flags"]
            )

            # Prepare results in unified format
            results = {}

            # Process landmarks
            for landmark_id, landmark_data in curr_landmarks.items():
                # Get current position
                x = int(landmark_data["x"])
                y = int(landmark_data["y"])

                # Get flow vector at this position if within bounds
                if 0 <= y < flow.shape[0] and 0 <= x < flow.shape[1]:
                    dx, dy = flow[y, x]

                    # Store position and flow vector
                    results[landmark_id] = {
                        "x": float(x),
                        "y": float(y),
                        "dx": float(dx),
                        "dy": float(dy)
                    }

            return results

        except Exception as e:
            print(f"Error in optical flow tracking: {str(e)}")
            return None

    def _points_tracker(self, frame, landmarks, region_size, frame_idx, reset_interval):
        """
        Track specific points around landmarks using Lucas-Kanade optical flow.
        Resets tracking after specified interval using MediaPipe landmarks.
        """
        try:
            # Convert frame to grayscale for tracking
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Check if reset should occur
            should_reset = frame_idx % reset_interval == 0

            # Prepare results
            results = {}

            # If it's a reset frame, use MediaPipe landmarks as-is
            if should_reset:
                print(f"Frame {frame_idx}: Resetting points tracker with MediaPipe landmarks")
                for landmark_id, landmark_data in landmarks.items():
                    if isinstance(landmark_id, (int, float)):
                        results[landmark_id] = {
                            "x": landmark_data["x"],
                            "y": landmark_data["y"]
                        }
            else:
                # Get previous frame key
                prev_frame_key = f"frame_{frame_idx - 1}"

                # Check if we have previous frame data for points tracker
                if prev_frame_key not in self._prev_frame_data:
                    print(f"No previous frame data for points tracker, using MediaPipe landmarks")
                    # Use MediaPipe landmarks as fallback
                    for landmark_id, landmark_data in landmarks.items():
                        if isinstance(landmark_id, (int, float)):
                            results[landmark_id] = {
                                "x": landmark_data["x"],
                                "y": landmark_data["y"]
                            }
                else:
                    # Get points from previous frame
                    prev_points = []
                    prev_ids = []

                    for landmark_id, data in self._prev_frame_data[prev_frame_key].items():
                        if isinstance(landmark_id, (int, float)):
                            prev_points.append([float(data["x"]), float(data["y"])])
                            prev_ids.append(landmark_id)

                    if not prev_points:
                        print("No valid previous points to track")
                        # Use MediaPipe landmarks as fallback
                        for landmark_id, landmark_data in landmarks.items():
                            if isinstance(landmark_id, (int, float)):
                                results[landmark_id] = {
                                    "x": landmark_data["x"],
                                    "y": landmark_data["y"]
                                }
                    else:
                        # Convert to numpy array for optical flow
                        prev_points = np.array(prev_points, dtype=np.float32).reshape(-1, 1, 2)

                        # Calculate optical flow
                        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
                            self._prev_frame_data["gray"],
                            curr_gray,
                            prev_points,
                            None,
                            winSize=(15, 15),
                            maxLevel=2,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                        )

                        # Store tracked points
                        for i, (point, landmark_id, st) in enumerate(zip(curr_points, prev_ids, status)):
                            if st[0] == 1:  # Point was successfully tracked
                                results[landmark_id] = {
                                    "x": float(point[0][0]),
                                    "y": float(point[0][1])
                                }
                            else:  # Tracking failed, use MediaPipe landmark
                                if landmark_id in landmarks:
                                    results[landmark_id] = {
                                        "x": landmarks[landmark_id]["x"],
                                        "y": landmarks[landmark_id]["y"]
                                    }

            # Store current frame data for next iteration
            self._prev_frame_data = {
                "frame_key": f"frame_{frame_idx}",
                "gray": curr_gray
            }

            # Store tracked points
            for landmark_id, data in results.items():
                if isinstance(landmark_id, (int, float)):
                    self._prev_frame_data[landmark_id] = {
                        "x": data["x"],
                        "y": data["y"]
                    }

            # Add reset flag
            results["reset_occurred"] = should_reset

            return results

        except Exception as e:
            print(f"Error in points tracking: {str(e)}")
            traceback.print_exc()
            return None

    def _apply_proxy_scale(self, image_np, proxy_scale):
        """Helper method to apply proxy scale to image."""
        if proxy_scale < 1.0:
            h, w = image_np.shape[:2]
            new_h, new_w = int(h * proxy_scale), int(w * proxy_scale)
            return cv2.resize(image_np, (new_w, new_h))
        return image_np

    def _draw_debug_visualization(self, image, tracking_data, show_region, region_size):
        """
        Enhanced debug visualization showing different types of tracking results.
        """
        debug_image = image.copy()

        # Draw MediaPipe landmarks if available
        if "mediapipe" in tracking_data:
            self._draw_landmarks(
                debug_image,
                tracking_data["mediapipe"],
                (0, 255, 0),  # Green color for MediaPipe
                2,  # Radius
                show_region,
                region_size
            )

        # Draw Optical Flow if available
        if "optical_flow" in tracking_data:
            self._draw_landmarks_with_flow(
                debug_image,
                tracking_data["optical_flow"],
                (255, 0, 0),  # Blue color for Optical Flow
                2  # Radius
            )

        # Draw Points Tracker if available
        if "points_tracker" in tracking_data:
            # Get points without the reset flag
            points_data = {k: v for k, v in tracking_data["points_tracker"].items()
                           if k != "reset_occurred"}

            self._draw_landmarks(
                debug_image,
                points_data,
                (0, 0, 255),  # Red color for Points Tracker
                2,  # Radius
                show_region,
                region_size
            )

            # Indicate if reset occurred
            if tracking_data["points_tracker"].get("reset_occurred", False):
                cv2.putText(
                    debug_image,
                    "Reset Occurred",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

        return debug_image

    def _draw_landmarks(self, image, landmarks_data, color, radius, show_region, region_size):
        """
        Helper method to draw landmarks on the image.
        """
        for landmark_id, data in landmarks_data.items():
            # Skip non-numeric keys (like "reset_occurred")
            if not isinstance(landmark_id, (int, float)):
                continue

            # Convert to int only for drawing
            x, y = int(data["x"]), int(data["y"])
            cv2.circle(image, (x, y), radius, color, -1)

            # Draw region if requested
            if show_region:
                half_size = region_size // 2
                x1 = max(0, x - half_size)
                y1 = max(0, y - half_size)
                x2 = min(image.shape[1], x + half_size)
                y2 = min(image.shape[0], y + half_size)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    def _draw_landmarks_with_flow(self, image, flow_data, color, radius):
        """
        Helper method to draw landmarks with flow vectors.
        """
        for landmark_id, data in flow_data.items():
            x, y = int(data["x"]), int(data["y"])

            # Draw landmark point
            cv2.circle(image, (x, y), radius, color, -1)

            # Draw flow vector if available
            if "dx" in data and "dy" in data:
                dx, dy = data["dx"], data["dy"]
                # Scale flow vector for visibility (optional)
                scale = 5.0
                end_x, end_y = int(x + dx * scale), int(y + dy * scale)
                cv2.arrowedLine(image, (x, y), (end_x, end_y), color, 1, cv2.LINE_AA, 0, 0.3)

    def _save_fp_pipe_state(self, fp_pipe, filename=None):
        """
        Save fp_pipe state to a file for debugging purposes.
        Args:
            fp_pipe: The fp_pipe dictionary to save
            filename: Optional filename, if None will generate based on timestamp
        """
        try:
            # Create debug directory if it doesn't exist
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'debug')
            os.makedirs(debug_dir, exist_ok=True)

            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"fp_pipe_state_{timestamp}.json"

            filepath = os.path.join(debug_dir, filename)

            # Convert numpy arrays and other non-serializable types to lists
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                return obj

            # Create a copy of fp_pipe and convert all values
            serializable_pipe = {}
            for key, value in fp_pipe.items():
                if isinstance(value, dict):
                    serializable_pipe[key] = {
                        k: convert_to_serializable(v)
                        for k, v in value.items()
                    }
                else:
                    serializable_pipe[key] = convert_to_serializable(value)

            # Save to file with pretty printing
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_pipe, f, indent=4)

            print(f"Saved fp_pipe state to: {filepath}")

        except Exception as e:
            print(f"Error saving fp_pipe state: {str(e)}")
            traceback.print_exc()