import jax
import jax.numpy as jnp
import numpy as np

from typing import List, Tuple

from rtdlf_preprocessor.util import write_progress
from rtdlf_preprocessor.video.camera import Camera
from rtdlf_preprocessor.video.depth_video import DepthVideo


"""Depth processor

This module contains functions to generate vertices from depth videos.
"""


def depth_to_vertex_matrices(
    cameras: List[Camera], depth_vids: List[DepthVideo], frame: int
) -> jnp.ndarray:
    """Depth frame to point cloud

    Get a point cloud of the scene from the given depth cameras.

    :param cameras: The camera metadata.
    :type cameras: List[Camera] (len l)
    :param depth_vids: The depth video's.
    :type depth_vids: List[DepthVideo]
    :param frame: The frame to generate the point cloud for.
    :type frame: int
    :returns: l vertex_matrices, one for each camera
    :rtype: jnp.ndarray of shape (l,n,m,3) with dtype float32
    """

    # get the required arguments
    depth_frames = jnp.array([vid.get_depth_frame(frame) for vid in depth_vids])
    models = jnp.array([cam.model for cam in cameras])
    ranges = jnp.array([cam.depth_range.as_array() for cam in cameras])
    pps = jnp.array([cam.pp.as_array() for cam in cameras])
    focals = jnp.array([cam.focal.as_array() for cam in cameras])

    width_arrays = jnp.array([jnp.arange(cam.resolution.width) for cam in cameras])
    height_arrays = jnp.array([jnp.arange(cam.resolution.height) for cam in cameras])

    # compute the vertices for each camera and flatten to list of vertices representing a point cloud
    return jax.vmap(depth_to_vertices)(
        depth_frames,
        models,
        ranges,
        pps,
        focals,
        width_arrays,
        height_arrays,
    )


@jax.jit
def depth_to_vertices(
    depth_frame: jnp.ndarray,
    model: jnp.ndarray,
    depth_range: jnp.ndarray,
    pp: jnp.ndarray,
    focal: jnp.ndarray,
    width_array: jnp.ndarray,
    height_array: jnp.ndarray,
) -> jnp.ndarray:
    """Compute a matrix of vertices, from a matrix of depth info of a single camera.

    :param depth_frame: The matrix with depth info. shape (n,m)
    :type depth_frame: jnp.ndarray

    :returns: A matrix of shape (n,m,3) with the vertices.
    :rtype: jnp.ndarray
    """  # compute the (x,y,z) components of each vertex
    zs = _z_vertex_positions(depth_frame, depth_range[0], depth_range[1])
    xs = _x_vertex_positions(zs, pp[0], focal[0], width_array)
    ys = _y_vertex_positions(zs, pp[1], focal[1], height_array)
    ones = jnp.ones(zs.shape)

    # merge components to actual vertices
    vertices = jnp.stack((xs, ys, zs, ones), axis=-1)

    @jax.jit
    def vertex_row_to_world(row: jnp.ndarray) -> jnp.ndarray:
        "Takes a row of untransformed vertices and transforms them to world space"
        return jax.vmap(lambda vertex: model @ vertex)(row)

    # Transform the vertices from camera to world space
    jax_vertices = jax.vmap(vertex_row_to_world)(vertices)

    # return the vertices without their trailing 1
    return jax_vertices[..., :3]


@jax.jit
def _x_vertex_positions(
    depth_matrix: jnp.ndarray, pp_x: float, focal_x: float, width_array: jnp.ndarray
) -> jnp.ndarray:
    return depth_matrix * ((width_array + 0.5 - pp_x) / focal_x)


@jax.jit
def _y_vertex_positions(
    depth_matrix: jnp.ndarray, pp_y: float, focal_y: float, height_array: jnp.ndarray
) -> jnp.ndarray:
    return depth_matrix * ((height_array + 0.5 - pp_y) / focal_y).reshape(-1, 1)


MAXDEPTH = 1023


@jax.jit
def _z_vertex_positions(
    depth_frame: jnp.ndarray, near: float, far: float
) -> jnp.ndarray:
    """Get the depth in meters as a matrix"""

    @jax.jit
    def depth_to_meters(dd: int) -> float:
        normalized_depth = dd / MAXDEPTH
        return 1 / (1 / far + normalized_depth * (1 / near - 1 / far))

    return jax.vmap(depth_to_meters)(depth_frame)


##############################
## Center vertex calculator ##
##############################


def get_center_coordinate(camera: Camera) -> Tuple[int, int]:
    return camera.resolution.width // 2, camera.resolution.height // 2


def get_cameras_centers(
    cameras: List[Camera], depth_vids: List[DepthVideo], max_frames
) -> np.ndarray:
    ret = np.empty((0, 3))
    for frame in range(max_frames):
        write_progress(frame + 1, max_frames, "Generating camera center vertices")
        for i in range(len(cameras)):
            cam = cameras[i]
            u, v = get_center_coordinate(cam)
            d = depth_vids[i].get_depth_frame(frame)
            z = 1 / (
                1 / cam.depth_range.far
                + d[u][v] * (1 / cam.depth_range.near - 1 / cam.depth_range.far)
            )
            x = (u + 0.5 - cam.pp.x) / cam.focal.x * z
            y = (v + 0.5 - cam.pp.y) / cam.focal.y * z
            p = np.array([x, y, z, 1])
            p = cam.model @ p
            ret = np.append(ret, [p[:3]], axis=0)
    ret = ret.astype(np.float32)
    return ret
