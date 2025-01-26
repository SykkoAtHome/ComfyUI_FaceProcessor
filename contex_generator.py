import os
from pathlib import Path
from typing import Set, Optional


def generate_context_file(
    root_dir: str,
    output_file: str,
    ignore_dirs: Optional[Set[str]] = None,
    ignore_files: Optional[Set[str]] = None,
) -> None:
    """
    Generate a single file containing all code with file paths as headers.

    Args:
        root_dir: Root directory to start scanning.
        output_file: Output file path.
        ignore_dirs: Set of directory names to ignore.
        ignore_files: Set of file names to ignore.
    """
    # Default ignored directories and files
    if ignore_dirs is None:
        ignore_dirs = {".git", "__pycache__", ".venv", "venv", ".idea", "node_modules"}
    if ignore_files is None:
        ignore_files = set()

    # Ensure the output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out:
        for root, dirs, files in os.walk(root_dir):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]

            # Process each file
            for file in files:
                if file in ignore_files:
                    continue

                # Check if the file has a supported extension
                if file.endswith((".py", ".sql", ".js", ".tsx", ".ts", ".jsx", ".json", ".css", ".html")):
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(root_dir)

                    # Write file path as a header
                    out.write(f"\n# {relative_path}\n")

                    # Write file content
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            out.write(f.read())
                            out.write("\n")
                    except Exception as e:
                        out.write(f"# Error reading file: {str(e)}\n")


# Example usage
if __name__ == "__main__":
    output_file = "D:/code/code_base/face_processor_05.txt"
    generate_context_file(
        root_dir="E:/ComfyUI-DEV/custom_nodes/ComfyUI_FaceProcessor",
        output_file=output_file,
        ignore_dirs={".git", "__pycache__", ".venv", "venv", ".idea", "node_modules", "models"},
        ignore_files={".gitignore", "contex_generator.py", output_file, "package-lock.json", "TODO.txt", "mediapipe_landmarks_468.obj"},
    )