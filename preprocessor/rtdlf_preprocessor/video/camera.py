import jax.numpy as jnp

from flax import struct


@struct.dataclass
class Resolution:
    width: int
    height: int

    def as_array(self) -> jnp.ndarray:
        return jnp.array([self.width, self.height], dtype=jnp.uint32)


@struct.dataclass
class Focal:
    x: float
    y: float

    def as_array(self) -> jnp.ndarray:
        return jnp.array([self.x, self.y], dtype=jnp.float32)


@struct.dataclass
class PrincipalPoint:
    x: float
    y: float

    def as_array(self) -> jnp.ndarray:
        return jnp.array([self.x, self.y], dtype=jnp.float32)


@struct.dataclass
class DepthRange:
    near: float
    far: float

    def as_array(self) -> jnp.ndarray:
        return jnp.array([self.near, self.far], dtype=jnp.float32)


@struct.dataclass
class Camera:
    """Class representing a single camera."""

    resolution: Resolution
    focal: Focal
    pp: PrincipalPoint
    depth_range: DepthRange

    model: jnp.ndarray
    position: jnp.ndarray

    camera_idx: int

    def as_array(self):
        """(camera_idx, resolution, focal, pp, depth_range, model, position)"""
        return (
            self.camera_idx,
            self.resolution,
            self.focal,
            self.pp,
            self.depth_range,
            self.model,
            self.position,
        )
