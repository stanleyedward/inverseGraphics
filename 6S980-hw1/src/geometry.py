import torch
from jaxtyping import Float
from torch import Tensor


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional points into (n+1)-dimensional homogeneous points."""
    # points = [dim1, dim2, 3]
    ones = torch.ones(
        *points.shape[:-1], 1, dtype=points.dtype, device=points.device
    )  # [dim1, dim2, 1]
    homogenized_points = torch.cat((points, ones), dim=-1)  # [dim1, dim2, 4]
    return homogenized_points

    raise NotImplementedError("This is your homework.")


def homogenize_vectors(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional vectors into (n+1)-dimensional homogeneous vectors."""
    zeros = torch.zeros(*points.shape[:-1], 1, dtype=points.dtype, device=points.device)
    homogenized_vectors = torch.cat((points, zeros), dim=-1)
    return homogenized_vectors

    raise NotImplementedError("This is your homework.")


def transform_rigid(
    xyz: Float[Tensor, "*#batch 4"],
    transform: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Apply a rigid-body transform to homogeneous points or vectors."""
    return torch.einsum("...i, ...ij ->...i", xyz, transform)
    raise NotImplementedError("This is your homework.")


def transform_world2cam(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D world coordinates to homogeneous
    3D camera coordinates.
    """

    return torch.einsum("...i, ...ij ->...i", xyz, cam2world)
    raise NotImplementedError("This is your homework.")


def transform_cam2world(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D camera coordinates to homogeneous
    3D world coordinates.
    """

    return torch.einsum("...i, ...ij ->...i", xyz, cam2world)
    raise NotImplementedError("This is your homework.")


def project(
    xyz: Float[Tensor, "*#batch 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 2"]:
    """Project homogenized 3D points in camera coordinates to pixel coordinates."""
    zeros = torch.zeros(
        *intrinsics.shape[:-1], 1, dtype=intrinsics.dtype, device=intrinsics.device
    )  # [#batch, 3, 1]
    intrinsics_0 = torch.cat((intrinsics, zeros), dim=-1)  # [#batch, 3, 4]

    homogeneous_proj = torch.einsum(
        "...ij,...j -> ...i", intrinsics_0, xyz
    )  # [batch, 3]
    px_coords = torch.div(
        homogeneous_proj[..., :2] / homogeneous_proj[..., -1]
    )  # [batch,2]

    return px_coords
    raise NotImplementedError("This is your homework.")
