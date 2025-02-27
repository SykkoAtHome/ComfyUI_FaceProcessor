import traceback

import cv2
import numpy as np
import pandas as pd
import torch


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
                "show_landmarks": ("BOOLEAN", {
                    "default": False
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "default": None
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT", "MASK", "MASK")
    RETURN_NAMES = ("image", "fp_pipe", "mask", "tracking_mask")
    FUNCTION = "track_face"
    CATEGORY = "Face Processor"

    def track_face(self, image, fp_pipe, use_optical_flow, use_track_points, proxy_scale, show_landmarks, mask=None):
        """
        Main tracking method processing all frames in the batch.
        Currently only MediaPipe detection is implemented.
        Optical flow and points tracking are placeholders for future implementation.

        Args:
            image: Input image tensor
            fp_pipe: Face processing pipeline data
            use_optical_flow: Whether to use optical flow for tracking (placeholder)
            use_track_points: Whether to use point tracking (placeholder)
            proxy_scale: Scale factor for proxy processing
            show_landmarks: Whether to draw landmarks on output images
            mask: Optional mask for limiting processing area

        Returns:
            tuple: (image, updated fp_pipe, mask, tracking_mask)
        """
        try:
            # Import tqdm for progress bar
            from tqdm import tqdm

            # Process all frames in the batch
            batch_size = image.shape[0] if len(image.shape) == 4 else 1

            # Process each frame with progress bar
            for frame_idx in tqdm(range(batch_size), desc="Tracking faces", unit="frame"):
                # Get current frame
                current_frame = image[frame_idx:frame_idx + 1] if batch_size > 1 else image
                frame_np = self.image_processor.convert_to_numpy(current_frame)

                if frame_np is None:
                    print(f"Error: Failed to convert frame {frame_idx}")
                    continue

                # Get current mask for this frame if available
                current_mask_np = None
                if mask is not None:
                    # Handle batch of masks
                    if len(mask.shape) == 4 and mask.shape[0] > 1 and frame_idx < mask.shape[0]:
                        current_mask = mask[frame_idx:frame_idx + 1]
                    else:
                        current_mask = mask

                    # Convert mask to numpy using ImageProcessor
                    current_mask_np = self.image_processor.convert_mask_to_numpy(current_mask)

                self._process_single_frame(
                    frame_np,
                    frame_idx,
                    fp_pipe,
                    use_optical_flow,
                    use_track_points,
                    proxy_scale,
                    current_mask_np
                )

            # Generate tracking masks from detected landmarks and filter with input mask if provided
            tracking_mask = self._generate_tracking_masks(image, fp_pipe, mask)

            # Update landmarks with mask status information only if input mask was provided
            if mask is not None:
                self._update_landmarks_mask_status(fp_pipe, tracking_mask)

            # Draw landmarks on output images if requested
            output_image = image
            if show_landmarks:
                output_image = self._draw_landmarks_on_images(image, fp_pipe)

            # Return results
            return output_image, fp_pipe, mask, tracking_mask

        except Exception as e:
            print(f"Error in track_face: {str(e)}")
            traceback.print_exc()
            # Return empty tracking mask in case of error
            empty_mask = torch.zeros((batch_size, image.shape[1], image.shape[2]), dtype=torch.float32)
            return image, fp_pipe, mask, empty_mask

    def _process_single_frame(self, frame_np, frame_idx, fp_pipe, use_optical_flow, use_track_points, proxy_scale,
                              mask_np=None):
        """
        Process a single frame applying MediaPipe detection.
        Optical flow and points tracking are just placeholders at this stage.

        Args:
            frame_np: Input frame as numpy array
            frame_idx: Index of the current frame
            fp_pipe: Face processing pipeline data
            use_optical_flow: Whether to use optical flow for tracking (placeholder)
            use_track_points: Whether to use point tracking (placeholder)
            proxy_scale: Scale factor for proxy processing
            mask_np: Optional mask for limiting processing area
        """
        frame_key = f"frame_{frame_idx}"

        # Ensure frame exists in fp_pipe
        if frame_key not in fp_pipe["frames"]:
            fp_pipe["frames"][frame_key] = {}

        # Initialize tracking data if needed
        if "tracking" not in fp_pipe["frames"][frame_key]:
            fp_pipe["frames"][frame_key]["tracking"] = {}

        # Add mask information if available
        if mask_np is not None:
            fp_pipe["frames"][frame_key]["tracking"]["mask_used"] = True

            if np.max(mask_np) > 0:  # Only if mask has non-zero values
                # Store mask coverage information
                fp_pipe["frames"][frame_key]["tracking"]["mask_coverage"] = float(np.mean(mask_np > 0))

        # Step 1: MediaPipe Detection (only actual implementation)
        mediapipe_results, blur_score = self._mediapipe_detector(frame_np, proxy_scale)

        # Store blur score in fp_pipe at the frame level
        fp_pipe["frames"][frame_key]["blur_score"] = float(blur_score)

        if mediapipe_results:
            fp_pipe["frames"][frame_key]["tracking"]["mediapipe"] = mediapipe_results
        else:
            # Quiet failure, will be reflected in progress bar
            return

        # Step 2: Optical Flow placeholder notification (if enabled)
        if use_optical_flow and frame_idx > 0:
            # Just add a placeholder key to indicate it would be used
            fp_pipe["frames"][frame_key]["tracking"]["optical_flow_placeholder"] = True

        # Step 3: Points Tracking placeholder notification (if enabled)
        if use_track_points:
            # Just add a placeholder key to indicate it would be used
            fp_pipe["frames"][frame_key]["tracking"]["points_tracker_placeholder"] = True

    def _mediapipe_detector(self, image_np, proxy_scale):
        """
        Detect facial landmarks using MediaPipe.
        Returns detection results in unified format where key is landmark ID and value is
        a dictionary with x, y coordinates and confidence.
        Also calculates and includes blur score of the image.
        """
        try:
            # Apply proxy scale if needed
            working_image = self._apply_proxy_scale(image_np, proxy_scale)

            # Calculate blur score using the normalized Laplacian method
            _, blur_score = self.image_processor.calculate_blur_score(working_image, method='gradient', normalize=True)

            landmarks_df = self.face_detector.detect_landmarks_mp(working_image)
            if landmarks_df is None:
                # Even if no face is detected, return blur score
                return None, blur_score

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
                    "in_mask": 1
                }

            return results, blur_score

        except Exception as e:
            print(f"Error in MediaPipe detection: {str(e)}")
            return None, 0.0

    def _apply_proxy_scale(self, image_np, proxy_scale):
        """Helper method to apply proxy scale to image."""
        if proxy_scale < 1.0:
            h, w = image_np.shape[:2]
            new_h, new_w = int(h * proxy_scale), int(w * proxy_scale)
            return cv2.resize(image_np, (new_w, new_h))
        return image_np

    def _generate_tracking_masks(self, image, fp_pipe, input_mask=None, coverage_threshold=0.4):
        """
        Generate masks from detected facial landmarks stored in fp_pipe.
        If input_mask is provided, return only the intersection with the generated mask.
        Interpolates masks that have significantly lower coverage than their neighbors.

        Args:
            image: Input image tensor to determine mask size
            fp_pipe: Face processing pipeline data containing detection results
            input_mask: Optional input mask to filter the results
            coverage_threshold: Threshold for mask coverage difference (0-1, relative to neighbors average)

        Returns:
            torch.Tensor: Batch of face masks in ComfyUI MASK format
        """
        # Import tqdm for progress bar
        from tqdm import tqdm

        # Get image dimensions and batch size
        if len(image.shape) == 4:  # Batch of images
            batch_size, height, width = image.shape[0], image.shape[1], image.shape[2]
        else:  # Single image
            batch_size, height, width = 1, image.shape[0], image.shape[1]

        # Create empty output tensor for result masks and tracking coverage
        result_masks = []
        coverage_values = []

        # Process each frame to generate initial masks with progress bar
        for frame_idx in tqdm(range(batch_size), desc="Generating masks", unit="mask"):
            # Get frame key
            frame_key = f"frame_{frame_idx}"

            # Generate face mask from landmarks
            face_mask_tensor = torch.zeros((height, width), dtype=torch.float32)

            # Check if we have landmarks for this frame
            if (frame_key in fp_pipe["frames"] and
                    "tracking" in fp_pipe["frames"][frame_key] and
                    "mediapipe" in fp_pipe["frames"][frame_key]["tracking"]):

                # Get MediaPipe landmarks data
                landmarks_data = fp_pipe["frames"][frame_key]["tracking"]["mediapipe"]

                # Convert to DataFrame format
                landmark_rows = []
                for landmark_id, coords in landmarks_data.items():
                    landmark_rows.append({
                        'x': coords['x'],
                        'y': coords['y'],
                        'index': landmark_id
                    })

                # Create DataFrame and generate mask
                if landmark_rows:
                    landmarks_df = pd.DataFrame(landmark_rows)
                    face_mask = ImageProcessor.generate_face_mask(
                        image_size=(width, height),
                        landmarks_df=landmarks_df
                    )
                    face_mask_tensor = torch.from_numpy(face_mask)

            # Apply input mask if available (AND operation)
            if input_mask is not None:
                # Get corresponding input mask for this frame
                if len(input_mask.shape) == 4:  # [B, 1, H, W]
                    frame_input_mask = input_mask[frame_idx][0] if frame_idx < input_mask.shape[0] else input_mask[-1][
                        0]
                elif len(input_mask.shape) == 3:  # [B, H, W] or [1, H, W]
                    if input_mask.shape[0] == 1:
                        frame_input_mask = input_mask[0]  # Single mask
                    else:
                        frame_input_mask = input_mask[frame_idx] if frame_idx < input_mask.shape[0] else input_mask[-1]
                else:  # [H, W]
                    frame_input_mask = input_mask

                # Perform AND operation (intersection)
                face_mask_tensor = face_mask_tensor * frame_input_mask

            # Calculate mask coverage (ratio of non-zero pixels)
            total_pixels = height * width
            mask_pixels = face_mask_tensor.sum().item()
            coverage = mask_pixels / total_pixels if total_pixels > 0 else 0

            # Store coverage for potential interpolation (only internal use, not stored in fp_pipe)
            coverage_values.append(coverage)

            # Add to results
            result_masks.append(face_mask_tensor)

        # Check for masks with significantly lower coverage than neighbors and interpolate
        if batch_size > 1:  # Need at least 2 frames for any kind of interpolation
            # Make a copy of the original masks to avoid modifying masks while iterating
            interpolated_masks = result_masks.copy()

            # Handle case with at least 3 frames (regular interpolation for middle frames)
            if batch_size > 2:
                print("Checking for frames requiring interpolation...")
                for frame_idx in range(1, batch_size - 1):
                    current_coverage = coverage_values[frame_idx]
                    prev_coverage = coverage_values[frame_idx - 1]
                    next_coverage = coverage_values[frame_idx + 1]

                    # Calculate average coverage of neighboring frames
                    neighbors_avg = (prev_coverage + next_coverage) / 2

                    # If current mask has significantly lower coverage than the average of its neighbors
                    if neighbors_avg > 0 and (current_coverage / neighbors_avg) < coverage_threshold:
                        print(
                            f"Frame {frame_idx}: Interpolating mask (coverage ratio: {(current_coverage / neighbors_avg):.3f})")

                        # Get previous and next masks
                        prev_mask = result_masks[frame_idx - 1].cpu().numpy()
                        next_mask = result_masks[frame_idx + 1].cpu().numpy()

                        # Interpolate mask
                        interpolated_mask = ImageProcessor.mask_interpolation(prev_mask, next_mask)

                        # Replace current mask with interpolated one
                        interpolated_masks[frame_idx] = torch.from_numpy(interpolated_mask)

                        # Log new coverage for debugging
                        new_coverage = float(interpolated_mask.sum() / total_pixels)
                        print(
                            f"  Frame {frame_idx}: Coverage improved from {current_coverage:.4f} to {new_coverage:.4f}")

            # Handle first frame (if it has significantly lower coverage than the second frame)
            if batch_size >= 2:
                # Compare first frame with second frame
                first_coverage = coverage_values[0]
                second_coverage = coverage_values[1]

                # If first frame has significantly lower coverage
                if second_coverage > 0 and (first_coverage / second_coverage) < coverage_threshold:
                    print(f"First frame: Fixing low coverage mask (ratio: {(first_coverage / second_coverage):.3f})")

                    # Simply reuse the second frame mask for the first frame
                    interpolated_masks[0] = result_masks[1].clone()

            # Handle last frame (if it has significantly lower coverage than the second-to-last frame)
            if batch_size >= 2:
                # Compare last frame with second-to-last frame
                last_coverage = coverage_values[-1]
                prev_coverage = coverage_values[-2]

                # If last frame has significantly lower coverage
                if prev_coverage > 0 and (last_coverage / prev_coverage) < coverage_threshold:
                    print(f"Last frame: Fixing low coverage mask (ratio: {(last_coverage / prev_coverage):.3f})")

                    # Simply reuse the second-to-last frame mask for the last frame
                    interpolated_masks[-1] = result_masks[-2].clone()

            # Update result masks with interpolated ones
            result_masks = interpolated_masks

        # Stack into batch
        return torch.stack(result_masks)

    def _update_landmarks_mask_status(self, fp_pipe, tracking_mask):
        """
        Update landmarks in fp_pipe with information whether they're inside or outside the mask.

        Args:
            fp_pipe: Face processing pipeline data
            tracking_mask: Generated mask tensor (B, H, W)
        """
        try:
            # Process each frame
            batch_size = tracking_mask.shape[0]

            for frame_idx in range(batch_size):
                frame_key = f"frame_{frame_idx}"

                # Skip if frame data not found
                if frame_key not in fp_pipe["frames"]:
                    continue

                frame_data = fp_pipe["frames"][frame_key]

                # Skip if no tracking or MediaPipe data
                if "tracking" not in frame_data or "mediapipe" not in frame_data["tracking"]:
                    continue

                # Get current frame mask
                current_mask = tracking_mask[frame_idx]

                # Get landmarks for this frame
                landmarks_data = frame_data["tracking"]["mediapipe"]

                # Remove old mask coverage data if exists
                if "mask_coverage" in frame_data["tracking"]:
                    del frame_data["tracking"]["mask_coverage"]

                if "mask_used" in frame_data["tracking"]:
                    del frame_data["tracking"]["mask_used"]

                # Check each landmark against the mask
                for landmark_id, coords in landmarks_data.items():
                    x, y = int(coords["x"]), int(coords["y"])

                    # Ensure coordinates are within mask bounds
                    h, w = current_mask.shape
                    if 0 <= x < w and 0 <= y < h:
                        # Check if the landmark is inside the mask (value > 0)
                        in_mask = bool(current_mask[y, x] > 0)

                        # Add mask status to landmark data
                        landmarks_data[landmark_id]["in_mask"] = int(in_mask)
                    else:
                        # Landmark is outside image bounds
                        landmarks_data[landmark_id]["in_mask"] = 0

        except Exception as e:
            print(f"Error updating landmarks mask status: {str(e)}")
            traceback.print_exc()

    def _draw_landmarks_on_images(self, images, fp_pipe):
        """
        Draw landmarks on the images, color-coded based on their in_mask status.
        Green for in_mask=1, red for in_mask=0.

        Args:
            images: Input image tensor (B,H,W,C) or (H,W,C)
            fp_pipe: Face processing pipeline data with landmarks

        Returns:
            torch.Tensor: Images with landmarks drawn
        """
        try:
            # Handle batch vs single image
            is_batch = len(images.shape) == 4
            batch_size = images.shape[0] if is_batch else 1

            # Process each frame
            results = []

            for frame_idx in range(batch_size):
                # Get current frame
                current_frame = images[frame_idx:frame_idx + 1] if is_batch else images

                # Convert to numpy using ImageProcessor
                frame_np = self.image_processor.convert_to_numpy(current_frame)

                if frame_np is None:
                    # Skip this frame if conversion failed
                    if is_batch:
                        results.append(images[frame_idx:frame_idx + 1])
                    else:
                        results.append(images)
                    continue

                height, width = frame_np.shape[:2]

                # Get landmarks for current frame
                frame_key = f"frame_{frame_idx}"
                if (frame_key in fp_pipe["frames"] and
                        "tracking" in fp_pipe["frames"][frame_key] and
                        "mediapipe" in fp_pipe["frames"][frame_key]["tracking"]):

                    landmarks_data = fp_pipe["frames"][frame_key]["tracking"]["mediapipe"]

                    # Separate landmarks by in_mask status
                    in_mask_landmarks = []
                    out_mask_landmarks = []

                    for landmark_id, coords in landmarks_data.items():
                        landmark_row = {
                            'x': coords['x'],
                            'y': coords['y'],
                            'index': landmark_id
                        }

                        # Add to appropriate list based on in_mask status
                        if coords.get('in_mask', 1) == 1:
                            in_mask_landmarks.append(landmark_row)
                        else:
                            out_mask_landmarks.append(landmark_row)

                    # Create DataFrames for each group
                    in_mask_df = pd.DataFrame(in_mask_landmarks) if in_mask_landmarks else None
                    out_mask_df = pd.DataFrame(out_mask_landmarks) if out_mask_landmarks else None

                    # Copy the original image
                    result_image = frame_np.copy()

                    # Draw landmarks directly on the image using OpenCV
                    if in_mask_df is not None and not in_mask_df.empty:
                        for _, landmark in in_mask_df.iterrows():
                            x, y = int(landmark['x']), int(landmark['y'])
                            if 0 <= x < width and 0 <= y < height:  # Ensure coordinates are within image bounds
                                # Draw filled green circle
                                cv2.circle(result_image, (x, y), 3, (0, 255, 0), -1, cv2.LINE_AA)

                    if out_mask_df is not None and not out_mask_df.empty:
                        for _, landmark in out_mask_df.iterrows():
                            x, y = int(landmark['x']), int(landmark['y'])
                            if 0 <= x < width and 0 <= y < height:  # Ensure coordinates are within image bounds
                                # Draw filled red circle
                                cv2.circle(result_image, (x, y), 3, (255, 0, 0), -1, cv2.LINE_AA)

                    # Convert back to tensor
                    frame_tensor = self.image_processor.numpy_to_tensor(result_image)
                    results.append(frame_tensor)
                else:
                    # No landmarks data, keep original
                    if is_batch:
                        results.append(images[frame_idx:frame_idx + 1])
                    else:
                        results.append(images)

            # Combine results
            if is_batch:
                return torch.cat(results, dim=0)
            else:
                return results[0]

        except Exception as e:
            print(f"Error drawing landmarks: {str(e)}")
            traceback.print_exc()
            return images