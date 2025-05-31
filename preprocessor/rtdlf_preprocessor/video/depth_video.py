import jax.numpy as jnp

import yuvio

from .video import Video


class DepthVideo(Video):
    def __init__(self, path: str, width: int, height: int) -> None:
        super().__init__(path, width, height)

    def get_depth_frame(self, frame: int) -> jnp.ndarray:
        """Get the y channel for a given frame"""
        return jnp.array(
            yuvio.imread(self.path, self.width, self.height, "yuv420p10le", frame).y,
            dtype=jnp.uint16,
        )
