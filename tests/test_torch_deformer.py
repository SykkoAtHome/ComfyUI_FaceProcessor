import pytest

torch = pytest.importorskip("torch")

from faceprocessor.deformer import TorchDeformer


def make_checker_image(size: int = 4) -> torch.Tensor:
    base = torch.zeros((1, size, size, 3), dtype=torch.float32)
    for y in range(size):
        for x in range(size):
            base[0, y, x] = torch.tensor([(x + y) / (2 * size), x / size, y / size])
    return base


def test_identity_warp_returns_same_image():
    image = make_checker_image(4)
    landmarks = torch.tensor(
        [[0, 0], [3, 0], [3, 3], [0, 3]], dtype=torch.float32
    )
    triangles = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)

    deformer = TorchDeformer()
    warped = deformer.warp(image, landmarks, landmarks, triangles)

    assert torch.allclose(warped, image, atol=1e-4)


def test_single_triangle_translation_moves_colors():
    image = torch.zeros((1, 5, 5, 3), dtype=torch.float32)
    image[:, 1:3, 1:3] = torch.tensor([[[1.0, 0.0, 0.0]]])

    src_landmarks = torch.tensor(
        [[1.0, 1.0], [3.0, 1.0], [1.0, 3.0]], dtype=torch.float32
    )
    dst_landmarks = torch.tensor(
        [[2.0, 1.0], [4.0, 1.0], [2.0, 3.0]], dtype=torch.float32
    )
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.long)

    deformer = TorchDeformer()
    warped = deformer.warp(image, src_landmarks, dst_landmarks, triangles)

    translated_patch = warped[:, 1:3, 2:4]
    assert torch.all(translated_patch[..., 0] > 0.9)
    assert torch.all(translated_patch[..., 1:] < 1e-3)
