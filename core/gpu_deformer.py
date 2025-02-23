import numpy as np
from PIL import Image
import torch
from ..core.base_mesh import MediapipeBaseLandmarks


class GPUDeformer:
    """
    Class for handling face warping operations on GPU using CUDA acceleration.
    """

    @staticmethod
    def warp_face(image, source_landmarks, target_landmarks):
        """
        Warp the face image using parallel GPU processing with reverse mapping.
        Now includes boundary triangles for complete image warping.

        Args:
            image: Input PIL Image or numpy array
            source_landmarks: Detected face landmarks
            target_landmarks: Target base landmarks

        Returns:
            PIL Image: Warped image
        """
        try:
            import cupy as cp
            print("Starting GPU warping process with boundary triangles...")

            # Convert and validate input data
            source_landmarks = np.asarray(source_landmarks, dtype=np.float32)
            target_landmarks = np.asarray(target_landmarks, dtype=np.float32)

            # Debug landmark ranges
            print("\nLandmark ranges:")
            print(f"Source landmarks - X: [{source_landmarks[:, 0].min():.1f}, {source_landmarks[:, 0].max():.1f}], "
                  f"Y: [{source_landmarks[:, 1].min():.1f}, {source_landmarks[:, 1].max():.1f}]")
            print(f"Target landmarks - X: [{target_landmarks[:, 0].min():.1f}, {target_landmarks[:, 0].max():.1f}], "
                  f"Y: [{target_landmarks[:, 1].min():.1f}, {target_landmarks[:, 1].max():.1f}]")

            # Convert image to numpy array if needed
            if isinstance(image, Image.Image):
                img_np = np.array(image, dtype=np.uint8)
            else:
                img_np = np.asarray(image, dtype=np.uint8)

            h, w = img_np.shape[:2]
            channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
            print(f"\nImage shape: {img_np.shape}")

            # Get triangulation including boundary triangles
            triangles, all_points = MediapipeBaseLandmarks.get_face_triangles(size=(w, h))
            print(f"Loaded {len(triangles)} triangles (including boundary triangles)")

            # Create complete source and target point arrays
            n_base_landmarks = len(source_landmarks)
            n_all_points = len(all_points)
            print(f"Base landmarks: {n_base_landmarks}, Total points: {n_all_points}")

            # Initialize complete arrays for source and target points
            source_points = np.zeros((n_all_points, 2), dtype=np.float32)
            target_points = np.zeros((n_all_points, 2), dtype=np.float32)

            # Fill in face landmarks
            n_base = 468  # Hardcode original landmark count
            source_points[:n_base] = source_landmarks[:n_base]
            target_points[:n_base] = target_landmarks[:n_base]

            # Boundary points remain identical
            source_points[n_base:] = all_points[n_base:]
            target_points[n_base:] = all_points[n_base:]

            # CUDA kernel for parallel triangle processing with reverse mapping
            cuda_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void process_triangles(
                const float* source_points,
                const float* target_points,
                const int* triangles,
                const unsigned char* input_image,
                unsigned char* output_image,
                const int num_triangles,
                const int width,
                const int height,
                const int channels
            ) {
                // Get triangle index
                const int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid >= num_triangles) return;

                // Get triangle indices
                const int idx0 = triangles[tid * 3];
                const int idx1 = triangles[tid * 3 + 1];
                const int idx2 = triangles[tid * 3 + 2];

                // Get source points
                const float src_x0 = source_points[idx0 * 2];
                const float src_y0 = source_points[idx0 * 2 + 1];
                const float src_x1 = source_points[idx1 * 2];
                const float src_y1 = source_points[idx1 * 2 + 1];
                const float src_x2 = source_points[idx2 * 2];
                const float src_y2 = source_points[idx2 * 2 + 1];

                // Get target points
                const float dst_x0 = target_points[idx0 * 2];
                const float dst_y0 = target_points[idx0 * 2 + 1];
                const float dst_x1 = target_points[idx1 * 2];
                const float dst_y1 = target_points[idx1 * 2 + 1];
                const float dst_x2 = target_points[idx2 * 2];
                const float dst_y2 = target_points[idx2 * 2 + 1];

                // Calculate target triangle area
                const float dst_area = (dst_x1 - dst_x0) * (dst_y2 - dst_y0) - 
                                     (dst_x2 - dst_x0) * (dst_y1 - dst_y0);

                const float abs_area = abs(dst_area);
                if (abs_area < 1e-6f) return;

                // Calculate target triangle bounding box with border check
                const int min_x = max(0, (int)min(min(dst_x0, dst_x1), dst_x2));
                const int min_y = max(0, (int)min(min(dst_y0, dst_y1), dst_y2));
                const int max_x = min(width - 1, (int)(max(max(dst_x0, dst_x1), dst_x2) + 0.5f));
                const int max_y = min(height - 1, (int)(max(max(dst_y0, dst_y1), dst_y2) + 0.5f));

                // Process each pixel in target triangle's bounding box
                for (int y = min_y; y <= max_y; y++) {
                    for (int x = min_x; x <= max_x; x++) {
                        // Calculate barycentric coordinates
                        float lambda1 = ((y - dst_y2) * (dst_x0 - dst_x2) + 
                                     (dst_x2 - x) * (dst_y0 - dst_y2)) / dst_area;
                        float lambda2 = ((y - dst_y0) * (dst_x1 - dst_x0) + 
                                     (dst_x0 - x) * (dst_y1 - dst_y0)) / dst_area;
                        float lambda0 = 1.0f - lambda1 - lambda2;

                        // Check if point is inside triangle with tolerance
                        const float eps = 1e-5f;
                        if (lambda0 >= -eps && lambda1 >= -eps && lambda2 >= -eps &&
                            lambda0 <= 1+eps && lambda1 <= 1+eps && lambda2 <= 1+eps) {

                            // Calculate source position using barycentric coordinates
                            float source_x = lambda0 * src_x0 + lambda1 * src_x1 + lambda2 * src_x2;
                            float source_y = lambda0 * src_y0 + lambda1 * src_y1 + lambda2 * src_y2;

                            // Round to nearest pixel
                            int sx = (int)(source_x + 0.5f);
                            int sy = (int)(source_y + 0.5f);

                            // Copy pixel if within bounds
                            if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                                for (int c = 0; c < channels; c++) {
                                    output_image[(y * width + x) * channels + c] = 
                                        input_image[(sy * width + sx) * channels + c];
                                }
                            }
                        }
                    }
                }
            }
            ''', 'process_triangles')

            # Move data to GPU
            source_points_gpu = cp.asarray(source_points.ravel())
            target_points_gpu = cp.asarray(target_points.ravel())
            triangles_gpu = cp.asarray(triangles.ravel().astype(np.int32))
            img_gpu = cp.asarray(img_np)
            output_gpu = cp.zeros_like(img_gpu)

            # Configure CUDA grid
            threadsPerBlock = 256
            blocksPerGrid = (len(triangles) + threadsPerBlock - 1) // threadsPerBlock
            print(f"CUDA config: {blocksPerGrid} blocks, {threadsPerBlock} threads per block")

            # Execute kernel
            print(f"Processing {len(triangles)} triangles on GPU...")
            cuda_kernel(
                (blocksPerGrid,), (threadsPerBlock,),
                (source_points_gpu, target_points_gpu, triangles_gpu,
                 img_gpu, output_gpu, len(triangles),
                 w, h, channels)
            )
            cp.cuda.stream.get_current_stream().synchronize()
            print("CUDA kernel execution completed")
            print("Flushing CUDA cache...")
            torch.cuda.empty_cache()

            # Transfer result back to CPU and convert to PIL
            output_np = cp.asnumpy(output_gpu)
            return Image.fromarray(output_np)

        except ImportError as e:
            print(f"CuPy not installed or CUDA not available: {str(e)}")
            return image
        except Exception as e:
            print(f"Error during GPU warping: {str(e)}")
            print(f"Error details: {str(e.__class__.__name__)}")
            import traceback
            traceback.print_exc()
            return image
