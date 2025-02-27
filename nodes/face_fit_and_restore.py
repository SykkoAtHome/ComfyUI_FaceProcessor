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
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT", "MASK", "INT")
    RETURN_NAMES = ("image", "fp_pipe", "mask", "bbox_size")
    FUNCTION = "process_image"
    CATEGORY = "Face Processor"

    def process_image(self, mode, workflow, output_mode, padding_percent=0.0, bbox_size="1024",
                      image=None, fp_pipe=None):
        """Process images in either Fit or Restore mode with unified fp_pipe structure."""
        # Import tqdm for progress bar
        from tqdm import tqdm

        # Validate inputs
        if image is None and (workflow == "single_image" or mode == "Restore"):
            print("Error: Image is required for single_image workflow and Restore mode")
            return None, fp_pipe, None, int(bbox_size)

        if workflow == "image_sequence" and mode == "Fit" and (not fp_pipe or "frames" not in fp_pipe):
            print("Error: Valid fp_pipe data is required for Fit mode in image_sequence workflow")
            return None, fp_pipe, None, int(bbox_size)

        if mode == "Restore" and (not fp_pipe or "frames" not in fp_pipe):
            print("Error: Valid fp_pipe with frames is required for Restore mode")
            return None, fp_pipe, None, int(bbox_size)

        # Initialize fp_pipe if None
        if fp_pipe is None:
            fp_pipe = {
                "padding_percent": padding_percent,
                "target_lm": {},
                "frames": {}
            }
        else:
            fp_pipe["padding_percent"] = padding_percent

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
                # Wyszukanie, ile ramek mamy w fp_pipe
                total_frames = len(fp_pipe["frames"])

                # Przygotowanie struktury do przetwarzania ramek
                if mode == "Restore":
                    frames_to_process = {
                        idx: None for idx in range(total_frames)
                    }
                else:  # Fit mode
                    frames_to_process = {}
                    for i in range(total_frames):
                        frame_key = f"frame_{i}"
                        if frame_key in fp_pipe["frames"]:
                            frames_to_process[i] = fp_pipe["frames"][frame_key].get("original_image_path")

                results = {}

                # Process each frame with progress bar
                for frame_idx in tqdm(range(total_frames), desc="Processing frames", unit="frame"):
                    # Handle frame image based on mode
                    if mode == "Fit":
                        frame_path = frames_to_process.get(frame_idx)
                        frame_image = None  # Let _process_frame load the image
                    else:  # Restore
                        # Handle potential batch input
                        if image is not None and len(image.shape) == 4 and image.shape[0] > 1:  # We have a batch
                            if frame_idx < image.shape[0]:
                                frame_image = image[frame_idx:frame_idx + 1]  # Keep the batch dimension
                            else:
                                print(f"Warning: Frame index {frame_idx} exceeds batch size {image.shape[0]}")
                                frame_image = None
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

                    # Store results
                    results[frame_idx] = result

                # Return based on output_mode
                if output_mode == "current_frame":
                    # W trybie current_frame, zwracamy pierwszą ramkę jako domyślną
                    # lub pozwalamy użytkownikowi określić, którą ramkę zwrócić
                    # poprzez dodatkowy parametr (to mogłoby być dodane w przyszłości)
                    default_frame_idx = 0
                    if default_frame_idx in results:
                        return results[default_frame_idx]
                    else:
                        print(f"Warning: Frame {default_frame_idx} not found in sequence")
                        return None, fp_pipe, None, int(bbox_size)
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
                # Verify image is not None when loading from path
                if original_path and image is None:
                    image = self.image_processor.load_image_from_path(original_path)
                    if image is None:
                        print(f"Warning: Failed to load image from {original_path}")
                        return None, fp_pipe, None, int(bbox_size)

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
                        "rotation_angle": frame_settings["rotation_angle"]
                    }

                    # Add original path for sequence processing
                    if original_path:
                        frame_data["original_image_path"] = original_path

                    # Add original frame index if available in fp_pipe
                    if frame_key in fp_pipe["frames"] and "original_frame_index" in fp_pipe["frames"][frame_key]:
                        frame_data["original_frame_index"] = fp_pipe["frames"][frame_key]["original_frame_index"]

                    # Store frame data
                    fp_pipe["frames"][frame_key] = frame_data

            else:  # Restore mode
                frame_settings = fp_pipe["frames"].get(frame_key)
                if frame_settings is None:
                    print(f"Error: No settings found for frame {frame_number}")
                    return None, fp_pipe, None, int(bbox_size)

                result = self._restore(image, frame_settings)
                if result is None:
                    print(f"Error: Restore operation failed for frame {frame_number}")
                    return None, fp_pipe, None, int(bbox_size)

                result_image, result_mask = result

            # Return results for both Fit and Restore modes
            if mode == "Fit":
                return result_image, fp_pipe, result_mask, int(bbox_size)
            else:
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
                print("Error: Failed to convert input image to numpy array")
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

            return restored_image, mask

        except Exception as e:
            print(f"Error in restore: {str(e)}")
            import traceback
            traceback.print_exc()
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
        try:
            if image is None:
                # Return a small default mask if image is None
                return torch.zeros((1, 64, 64), dtype=torch.float32)

            if torch.is_tensor(image):
                shape = image.shape[1:3]
            else:
                shape = image.shape[:2]
            return torch.zeros((1, *shape), dtype=torch.float32)
        except Exception as e:
            # Fallback to a default size if any error occurs
            print(f"Error creating empty mask: {str(e)}")
            return torch.zeros((1, 64, 64), dtype=torch.float32)