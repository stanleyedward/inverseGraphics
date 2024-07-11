import torch
from geometry import homogenize_points, transform_world2cam, project
from jaxtyping import Float
from torch import Tensor


def render_point_cloud(
    vertices: Float[Tensor, "vertex 3"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> Float[Tensor, "batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """
    batch_size = extrinsics.shape[0]
    height, width = resolution
    
    canvas = torch.ones((batch_size, height, width), dtype=torch.float32)
    
    homogenized_vertices = homogenize_points(vertices)
    camera_coords = transform_world2cam(homogenized_vertices, extrinsics)
    image_coords = project(camera_coords, intrinsics)
    
    


    raise NotImplementedError("This is your homework.")
