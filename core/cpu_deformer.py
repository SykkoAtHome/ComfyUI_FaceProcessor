import numpy as np
from PIL import Image
from ..core.base_landmarks import MediapipeBaseLandmarks

class CPUDeformer:
    """
    Class for handling face warping operations on CPU using triangle-based warping.
    """

    @staticmethod
    def warp_face(image, source_landmarks, target_landmarks):
        """
        Warp the face image to match the base landmarks using triangle-based warping on CPU.
        Now includes boundary triangles for complete image warping.

        Args:
            image: PIL Image to warp
            source_landmarks: Source face landmarks
            target_landmarks: Target face landmarks

        Returns:
            PIL Image: Warped image
        """
        try:
            # Get image dimensions
            w, h = image.size
            print(f"Image size: {w}x{h}")

            # Convert landmarks to numpy arrays if they aren't already
            source_landmarks = np.array(source_landmarks)
            target_landmarks = np.array(target_landmarks)

            # Get triangulation and all landmarks (including boundary points)
            print("Getting triangulation with boundary triangles...")
            triangles, all_target_points = MediapipeBaseLandmarks.get_face_triangles(size=(w, h))

            # Create arrays for source and target points
            n_base_landmarks = len(source_landmarks)
            n_all_points = len(all_target_points)
            n_boundary_points = n_all_points - n_base_landmarks

            print(f"Base landmarks: {n_base_landmarks}, Total points: {n_all_points}")

            # Create complete source and target point arrays
            source_points = np.zeros((n_all_points, 2), dtype=np.float32)
            target_points = np.zeros((n_all_points, 2), dtype=np.float32)

            # Fill in face landmarks
            source_points[:n_base_landmarks] = source_landmarks
            target_points[:n_base_landmarks] = target_landmarks

            # Fill in boundary points (use the same points for both source and target)
            if n_boundary_points > 0:
                boundary_points = all_target_points[n_base_landmarks:]
                source_points[n_base_landmarks:] = boundary_points
                target_points[n_base_landmarks:] = boundary_points

            print(f"Processing {len(triangles)} triangles")

            # Convert PIL image to numpy array
            img_np = np.array(image)
            output = np.zeros_like(img_np)

            # Process each triangle
            for i, triangle in enumerate(triangles):
                # Get triangle vertices
                src_tri = source_points[triangle]
                dst_tri = target_points[triangle]

                # Calculate bounding box for target triangle
                min_x = max(0, int(np.min(dst_tri[:, 0])))
                min_y = max(0, int(np.min(dst_tri[:, 1])))
                max_x = min(w - 1, int(np.ceil(np.max(dst_tri[:, 0]))))
                max_y = min(h - 1, int(np.ceil(np.max(dst_tri[:, 1]))))

                # Skip if triangle is outside image bounds
                if min_x >= max_x or min_y >= max_y:
                    continue

                # Calculate target triangle matrix for barycentric coordinates
                dst_matrix = np.vstack([
                    dst_tri[1] - dst_tri[0],
                    dst_tri[2] - dst_tri[0]
                ]).T

                # Calculate inverse matrix for barycentric coordinates if possible
                try:
                    dst_matrix_inv = np.linalg.inv(dst_matrix)
                except np.linalg.LinAlgError:
                    continue  # Skip degenerate triangles

                # Process each pixel in target triangle's bounding box
                for y in range(min_y, max_y + 1):
                    for x in range(min_x, max_x + 1):
                        # Calculate barycentric coordinates
                        point = np.array([x, y]) - dst_tri[0]
                        bary = np.dot(dst_matrix_inv, point)
                        alpha = 1.0 - bary[0] - bary[1]

                        # Check if point is inside triangle
                        eps = 1e-5
                        if (alpha >= -eps and bary[0] >= -eps and bary[1] >= -eps and
                                alpha <= 1 + eps and bary[0] <= 1 + eps and bary[1] <= 1 + eps):

                            # Calculate source pixel position using barycentric coordinates
                            src_x = int(alpha * src_tri[0, 0] + bary[0] * src_tri[1, 0] +
                                      bary[1] * src_tri[2, 0] + 0.5)
                            src_y = int(alpha * src_tri[0, 1] + bary[0] * src_tri[1, 1] +
                                      bary[1] * src_tri[2, 1] + 0.5)

                            # Copy pixel if within bounds
                            if (0 <= src_x < w and 0 <= src_y < h):
                                output[y, x] = img_np[src_y, src_x]

                # Print progress every 10%
                if (i + 1) % max(1, len(triangles) // 10) == 0:
                    print(f"Processed {i + 1}/{len(triangles)} triangles ({(i + 1) * 100 / len(triangles):.1f}%)")

            return Image.fromarray(output)

        except Exception as e:
            print(f"Error during CPU warping: {str(e)}")
            import traceback
            traceback.print_exc()
            return image