import torch
from jaxtyping import Float
from torch import Tensor
from src.geometry import homogenize_points, transform_world2cam, project


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
    image_coords = project(camera_coords, intrinsics) #[batch, 2]
    
    #scale to px coords
    
    image_coords[..., 0] *=width/2
    image_coords[..., 1] *=height/2
    image_coords[..., 0] +=width/2
    image_coords[..., 1] +=height/2
    
    px_coords = image_coords.round().to(torch.int64)
    
    mask = (px_coords[..., 0] >= 0) & (px_coords[..., 0] < width) & \
       (px_coords[..., 1] >= 0) & (px_coords[..., 1] < height)
       
    for b in range(batch_size):
        valid_coords = px_coords[b, mask[b]]
        canvas[b, valid_coords[:,1], valid_coords[:, 0]] = 0.0
    
    return canvas

    raise NotImplementedError("This is your homework.")
