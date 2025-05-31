import jax
import jax.numpy as jnp
import numpy as np

from typing import List, Optional, Tuple

from rtdlf_preprocessor.vertex.depth_processor import depth_to_vertex_matrices
from rtdlf_preprocessor.video.camera import Camera
from rtdlf_preprocessor.video.color_video import ColorVideo
from rtdlf_preprocessor.video.depth_video import DepthVideo


def point_cloud(
    cameras: List[Camera],
    depth_vids: List[DepthVideo],
    frame: int,
) -> jnp.ndarray:
    return depth_to_vertex_matrices(cameras, depth_vids, frame)


def voxel_cloud(
    cameras: List[Camera],
    depth_vids: List[DepthVideo],
    frame: int,
    voxel_size: float = 0.01,
) -> jnp.ndarray:
    """Get a point cloud of a frame in a voxel grid"""
    vertex_matrices = depth_to_vertex_matrices(cameras, depth_vids, frame)
    # get point cloud list
    pcl = vertex_matrices.reshape(-1, 3)

    # snap to voxel grid
    discrete_pcl = jnp.round(pcl / voxel_size).astype(
        jnp.int32
    )  # divide by grid size and floor, to snap to grid
    unique_discrete_pcl = jnp.unique(discrete_pcl, axis=0)  # remove duplicates
    return unique_discrete_pcl * voxel_size  # multiply to restore original space


def voxel_color_cloud(
    cameras: List[Camera],
    depth_vids: List[DepthVideo],
    color_vids: List[ColorVideo],
    frame: int,
    voxel_size: float = 0.01,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get a point cloud of a frame in a voxel grid"""
    vertex_matrices = depth_to_vertex_matrices(cameras, depth_vids, frame)
    color_matrics = jnp.stack(
        [vid.get_yuv_interleaved_frame(frame).astype(jnp.uint8) for vid in color_vids],
        axis=0,
    )

    # get point cloud list
    pcl = vertex_matrices.reshape(-1, 3)
    colors_flat = color_matrics.reshape(-1, 3)

    # snap to voxel grid
    vcl = jnp.floor(pcl / voxel_size).astype(
        jnp.int32
    )  # divide by grid size and floor, to snap to grid
    unique_vcl, inverse_indices = jnp.unique(
        vcl, axis=0, return_inverse=True
    )  # remove duplicates

    # Average colors for each voxel
    num_voxels = unique_vcl.shape[0]
    color_sums = jnp.zeros((num_voxels, 3), dtype=jnp.float32)
    color_sums = color_sums.at[inverse_indices].add(colors_flat)  # sum the colors

    counts = jnp.zeros((num_voxels,), dtype=jnp.float32)
    counts = counts.at[inverse_indices].add(
        1.0
    )  # get amount of values that were summed

    mean_colors = color_sums / counts[:, None]  # take average
    mean_colors = mean_colors.astype(jnp.uint8)

    return (
        unique_vcl * voxel_size,  # multiply to restore original space
        mean_colors,
    )
