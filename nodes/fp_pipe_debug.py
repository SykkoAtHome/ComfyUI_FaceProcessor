import copy
import traceback


class FacePipeDebug:
    """ComfyUI node for debugging and modifying face processor pipe data."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fp_pipe": ("DICT",),
                "hide_tracking": ("BOOLEAN", {
                    "default": False
                }),
                "limit_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "description": "Limit number of frames in fp_pipe (0 = keep all)"
                }),
                "console_output": ("BOOLEAN", {
                    "default": True,
                    "description": "Enable/disable console output"
                })
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("fp_pipe",)
    FUNCTION = "debug_pipe"
    CATEGORY = "Face Processor/Tools"

    def debug_pipe(self, fp_pipe, hide_tracking, limit_frames, console_output):
        """
        Modify face processor pipe data by hiding tracking data and limiting frames.

        Args:
            fp_pipe: Face processor pipe data dictionary
            hide_tracking: Whether to hide tracking data to reduce clutter
            limit_frames: Limit number of frames in fp_pipe (0 = keep all)
            console_output: Whether to print information to console

        Returns:
            dict: Modified fp_pipe data
        """
        try:
            # Create a deep copy to avoid modifying the original
            modified_pipe = copy.deepcopy(fp_pipe)

            # Print basic info about the pipe if console output is enabled
            num_frames = len(modified_pipe.get("frames", {}))

            if console_output:
                print(f"FP_PIPE: {num_frames} frames")

            # Apply modifications if needed
            if hide_tracking or limit_frames > 0:
                # Apply frame limit if needed
                if limit_frames > 0 and num_frames > limit_frames:
                    # Get frame keys and sort them numerically by frame number
                    frame_keys = list(modified_pipe.get("frames", {}).keys())

                    # Sort frames by numeric index (extract numbers from frame_X keys)
                    def get_frame_number(key):
                        try:
                            return int(key.split('_')[1])
                        except (IndexError, ValueError):
                            return 0

                    # Sort by numeric frame index
                    frame_keys.sort(key=get_frame_number)

                    # Keep only the first limit_frames frames
                    frames_to_keep = frame_keys[:limit_frames]

                    # Create a new frames dictionary with only the kept frames
                    new_frames = {k: modified_pipe["frames"][k] for k in frames_to_keep}
                    modified_pipe["frames"] = new_frames

                    if console_output:
                        print(f"Limited to {limit_frames} frames")

                # Hide tracking data if requested
                if hide_tracking:
                    for frame_key, frame_data in modified_pipe.get("frames", {}).items():
                        if "tracking" in frame_data:
                            # Remove tracking data to save memory and reduce clutter
                            del frame_data["tracking"]

                    if console_output:
                        print("Tracking data removed")

            return (modified_pipe,)

        except Exception as e:
            if console_output:
                print(f"Error in FacePipeDebug: {str(e)}")
                traceback.print_exc()
            return (fp_pipe,)
