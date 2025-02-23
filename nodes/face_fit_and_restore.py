import cv2
import numpy as np
import torch

from ..core.face_detector import FaceDetector
from ..core.image_processor import ImageProcessor


class FaceFitAndRestore:
    """ComfyUI node for processing face images in Fit or Restore mode."""

    def __init__(self):
        self.face_detector = FaceDetector()
        self.image_processor = ImageProcessor()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow": (["single_image", "image_sequence"], {
                    "default": "single_image"
                }),
                "mode": (["Fit", "Restore"], {
                    "default": "Fit"
                }),
                "output_mode": (["current_frame", "batch_sequence"], {
                    "default": "current_frame"
                }),
                "padding_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "bbox_size": (["512", "1024", "2048"], {
                    "default": "1024"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "fp_pipe": ("DICT", {
                    "default": None
                }),
                "image_sequence": ("DICT", {
                    "default": None
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT", "MASK", "INT")
    RETURN_NAMES = ("image", "fp_pipe", "mask", "bbox_size")
    FUNCTION = "process_image"
    CATEGORY = "Face Processor"

    def process_image(self, mode, workflow, output_mode, padding_percent=0.0, bbox_size="1024",
                      image=None, fp_pipe=None, image_sequence=None):
        """Process images in either Fit or Restore mode with unified fp_pipe structure."""

        # Validate inputs
        if image is None and (workflow == "single_image" or mode == "Restore"):
            print("Error: Image is required for single_image workflow and Restore mode")
            return None, fp_pipe, None, int(bbox_size)

        if workflow == "image_sequence" and mode == "Fit" and (not image_sequence or "frames" not in image_sequence):
            print("Error: Valid image sequence data is required for Fit mode")
            return None, fp_pipe, None, int(bbox_size)

        if mode == "Restore" and (not fp_pipe or "frames" not in fp_pipe):
            print("Error: Valid fp_pipe with frames is required for Restore mode")
            return None, fp_pipe, None, int(bbox_size)

        # Initialize fp_pipe if None
        if fp_pipe is None:
            fp_pipe = {
                "workflow": workflow,
                "current_frame": 0,
                "padding_percent": padding_percent,
                "target_lm": {},
                "frames": {}
            }

        try:
            if workflow == "single_image":
                return self._process_frame(
                    mode=mode,
                    image=image,
                    frame_number=0,
                    padding_percent=padding_percent,
                    bbox_size=bbox_size,
                    fp_pipe=fp_pipe
                )
            else:  # image_sequence
                current_frame = (fp_pipe.get("current_frame", 0) if mode == "Restore"
                                 else image_sequence.get("current_frame", 0))
                fp_pipe["current_frame"] = current_frame

                if mode == "Restore":
                    total_frames = len(fp_pipe["frames"])
                    frames_to_process = {
                        idx: None for idx in range(total_frames)
                    }
                else:  # Fit mode
                    total_frames = len(image_sequence["frames"])
                    frames_to_process = image_sequence["frames"]

                print(f"Processing sequence of {total_frames} frames...")
                results = {}
                current_frame_result = None

                # Process each frame
                for frame_idx in range(total_frames):
                    print(f"Processing frame {frame_idx + 1}/{total_frames}")

                    if mode == "Fit":
                        frame_path = frames_to_process[frame_idx]
                        frame_image = self.image_processor.load_image_from_path(frame_path)
                        if frame_image is None:
                            print(f"Warning: Could not load frame {frame_idx}")
                            continue
                    else:  # Restore
                        # Handle potential batch input
                        if len(image.shape) == 4 and image.shape[0] > 1:  # We have a batch
                            if frame_idx < image.shape[0]:
                                frame_image = image[frame_idx:frame_idx + 1]  # Keep the batch dimension
                            else:
                                print(f"Warning: Frame index {frame_idx} exceeds batch size {image.shape[0]}")
                                continue
                        else:  # Single image
                            frame_image = image
                        frame_path = None

                    # Process frame
                    result = self._process_frame(
                        mode=mode,
                        image=frame_image,
                        frame_number=frame_idx,
                        padding_percent=padding_percent,
                        bbox_size=bbox_size,
                        fp_pipe=fp_pipe,
                        original_path=frame_path if mode == "Fit" else None
                    )

                    # Store results for both current frame and batch processing
                    results[frame_idx] = result
                    if frame_idx == current_frame:
                        current_frame_result = result

                # Return based on output mode
                if output_mode == "current_frame":
                    if current_frame_result is None:
                        print(f"Warning: Frame {current_frame} not found in sequence")
                        return None, fp_pipe, None, int(bbox_size)
                    return current_frame_result
                else:  # batch_sequence mode
                    # Collect all processed frames
                    all_frames = []
                    all_masks = []

                    for idx in range(total_frames):
                        frame_result = results.get(idx)
                        if frame_result is not None:
                            processed_image = frame_result[0]
                            if processed_image is not None:
                                all_frames.append(processed_image)
                                if frame_result[2] is not None:  # If mask exists
                                    all_masks.append(frame_result[2])

                    if not all_frames:
                        print("Warning: No frames were successfully processed")
                        return None, fp_pipe, None, int(bbox_size)

                    # Stack all frames into a single batch tensor
                    batched_frames = torch.cat(all_frames, dim=0)

                    # Stack masks if they exist
                    batched_masks = torch.cat(all_masks, dim=0) if all_masks else None

                    return batched_frames, fp_pipe, batched_masks, int(bbox_size)

        except Exception as e:
            print(f"Error in process_image: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, fp_pipe, None, int(bbox_size)

    def _process_frame(self, mode, image, frame_number, padding_percent, bbox_size, fp_pipe, original_path=None):
        """Process a single frame in either Fit or Restore mode."""
        try:
            frame_key = f"frame_{frame_number}"

            if mode == "Fit":
                # Process frame in Fit mode
                result_image, frame_settings, result_mask = self._fit(
                    image=image,
                    padding_percent=padding_percent,
                    bbox_size=bbox_size
                )

                if frame_settings:
                    # Create standardized frame data structure
                    frame_data = {
                        "bbox_size": int(bbox_size),
                        "crop_bbox": frame_settings["crop_bbox"],
                        "original_image_shape": frame_settings["original_image_shape"],
                        "rotation_angle": frame_settings["rotation_angle"],
                        "detected_lm": {}  # Will be populated by face_wrapper
                    }

                    # Add original path for sequence processing
                    if original_path:
                        frame_data["original_image_path"] = original_path

                    # Store frame data
                    fp_pipe["frames"][frame_key] = frame_data

            else:  # Restore mode
                frame_settings = fp_pipe["frames"].get(frame_key)
                if frame_settings is None:
                    print(f"Error: No settings found for frame {frame_number}")
                    return None, fp_pipe, None, int(bbox_size)

                result = self._restore(image, frame_settings)
                if result is None:
                    return None, fp_pipe, None, int(bbox_size)

                result_image, result_mask = result[0], result[2]

            return result_image, fp_pipe, result_mask, int(bbox_size)

        except Exception as e:
            print(f"Error processing frame {frame_number}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, fp_pipe, None, int(bbox_size)

    def _fit(self, image, padding_percent, bbox_size):
        """Fit mode: Crop and process the face."""
        # Convert image to numpy array
        image_np = self.image_processor.convert_to_numpy(image)
        if image_np is None:
            return image, {}, self._create_empty_mask(image)

        # Detect facial landmarks
        landmarks_df = self.face_detector.detect_landmarks_mp(image_np)
        if landmarks_df is None:
            print("No face detected, returning original image")
            return image, {}, self._create_empty_mask(image)

        # Calculate rotation and rotate image
        rotation_angle = self.image_processor.calculate_rotation_angle(landmarks_df)
        rotated_image, updated_landmarks = self.image_processor.rotate_image(image_np, landmarks_df)
        if rotated_image is None:
            print("Failed to rotate image, returning original")
            return image, {}, self._create_empty_mask(image)

        # Crop face region
        cropped_face, crop_bbox = self.image_processor.crop_face_to_square(
            rotated_image, updated_landmarks, padding_percent
        )
        if cropped_face is None:
            print("Failed to crop face, returning original image")
            return image, {}, self._create_empty_mask(image)

        # Resize to target size
        target_size = int(bbox_size)
        final_image = self.image_processor.resize_image(cropped_face, target_size)
        if final_image is None:
            print("Failed to resize image, returning original")
            return image, {}, self._create_empty_mask(image)

        # Convert to tensor
        final_image = self.image_processor.numpy_to_tensor(final_image)

        # Create frame settings
        frame_settings = {
            "original_image_shape": image_np.shape,
            "rotation_angle": rotation_angle,
            "crop_bbox": crop_bbox,
            "bbox_size": target_size
        }

        # Create mask
        mask = self._create_mask(
            image_shape=frame_settings["original_image_shape"],
            crop_bbox=frame_settings["crop_bbox"],
            rotation_angle=frame_settings["rotation_angle"]
        )

        return final_image, frame_settings, mask

    def _restore(self, image, frame_settings):
        """Restore mode: Restore the face back to original position."""
        try:
            # Validate frame settings
            required_keys = ["original_image_shape", "rotation_angle", "crop_bbox"]
            if not all(key in frame_settings for key in required_keys):
                print("Error: Missing required frame settings")
                return None

            # Convert input image to numpy
            processed_face_np = self.image_processor.convert_to_numpy(image)
            if processed_face_np is None:
                return None

            # Get settings
            original_shape = frame_settings["original_image_shape"]
            rotation_angle = frame_settings["rotation_angle"]
            x1, y1, w, h = frame_settings["crop_bbox"]

            # Create output image
            restored_image = np.zeros(original_shape, dtype=np.uint8)

            # Resize face to original crop size
            resized_face = cv2.resize(processed_face_np, (w, h), interpolation=cv2.INTER_LANCZOS4)

            # Place face in position
            restored_image[y1:y1 + h, x1:x1 + w] = resized_face

            # Rotate back if needed
            if rotation_angle != 0:
                height, width = original_shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
                restored_image = cv2.warpAffine(
                    restored_image,
                    rotation_matrix,
                    (width, height),
                    flags=cv2.INTER_LANCZOS4
                )

            # Convert to tensor
            restored_image = self.image_processor.numpy_to_tensor(restored_image)

            # Create mask
            mask = self._create_mask(original_shape, (x1, y1, w, h), rotation_angle)

            return restored_image, frame_settings, mask

        except Exception as e:
            print(f"Error in restore: {str(e)}")
            return None

    def _create_mask(self, image_shape, crop_bbox, rotation_angle):
        """Create a mask for the face area."""
        mask = np.zeros(image_shape[:2], dtype=np.float32)
        x1, y1, w, h = crop_bbox

        # Create face area mask
        mask[y1:y1 + h, x1:x1 + w] = 1.0

        # Rotate mask if needed
        if rotation_angle != 0:
            height, width = image_shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
            mask = cv2.warpAffine(
                mask,
                rotation_matrix,
                (width, height),
                flags=cv2.INTER_LINEAR
            )

        return self.image_processor.convert_mask_to_tensor(mask)

    def _create_empty_mask(self, image):
        """Create an empty mask matching the image dimensions."""
        if torch.is_tensor(image):
            shape = image.shape[1:3]
        else:
            shape = image.shape[:2]
        return torch.zeros((1, *shape), dtype=torch.float32)
