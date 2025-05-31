import numpy as np
import jax.numpy as jnp
import cv2 as cv

from enum import Enum
from typing import List
from pathlib import Path

from .generator import Generator
from rtdlf_preprocessor.util import write_progress
from rtdlf_preprocessor.video.camera import Camera
from rtdlf_preprocessor.video.color_video import ColorVideo
from rtdlf_preprocessor.writers import write_bytes


class FragmentFormat(Enum):
    RAW = "RAW"
    JPEG = "JPEG"
    YUV = "YUV"


class FragmentGenerator(Generator):
    _color_videos: List[ColorVideo]  # color videos

    def __init__(
        self,
        out_dir: Path,
        cameras: List[Camera],
        max_frames: int,
        color_videos: List[ColorVideo],
    ) -> None:
        super().__init__(out_dir, cameras, max_frames)
        self._color_videos = color_videos

    def generate(self, format: FragmentFormat, *args, **kwargs) -> None:
        """Generate the requested vertex format and save it to file"""
        for frame in range(self._max_frames):
            write_progress(
                frame + 1, self._max_frames, f"Generating color representation {format}"
            )
            match format:
                case FragmentFormat.RAW:
                    self._generate_raw_frame(frame, *args, **kwargs)
                case FragmentFormat.JPEG:
                    self._generate_jpeg_frame(frame, *args, **kwargs)
                case FragmentFormat.YUV:
                    self._generate_yuv_frame(frame, *args, **kwargs)
        print(" Done")

    def _yuv_bytes_texture_format(self, frame: int) -> bytes:
        data = np.concatenate(
            [vid.get_flat_yuv420_frame(frame) for vid in self._color_videos]
        )
        return data.tobytes()

    def _frame_to_texture_format_2D(self, frame: int) -> jnp.ndarray:
        """For a given frame number, get that frame of each camera and group them into a 2D array.
        This array contains all the flattened frames in a row. This format is what
        an opengl texture is expected to look like.

        :param frame: The frame to generate as texture format
        :type frame: int
        """
        return jnp.concatenate(
            [vid.get_frame(frame).astype(jnp.uint8) for vid in self._color_videos],
            axis=0,
        )

    def _frame_to_texture_format(self, frame: int) -> jnp.ndarray:
        """For a given frame number, get that frame of each camera and group them into a 1D array.
        This array contains all the flattened frames in a row. This format is what
        an opengl texture is expected to look like.

        :param frame: The frame to generate as texture format
        :type frame: int
        """
        return self._frame_to_texture_format_2D(frame).ravel()

    def _generate_raw_frame(self, frame: int) -> None:
        out_path: str = str((self._get_frame_dir(frame) / "colors.raw").resolve())
        self._threadpool.submit_task(
            write_bytes,
            out_path=out_path,
            data=self._frame_to_texture_format(frame).tobytes(),
        )

    def _generate_jpeg_frame(self, frame: int) -> None:
        out_path: str = str((self._get_frame_dir(frame) / "colors.jpeg").resolve())
        self._threadpool.submit_task(
            cv.imwrite,
            filename=out_path,
            img=np.array(self._frame_to_texture_format_2D(frame)),
        )

    def _generate_yuv_frame(self, frame: int) -> None:
        out_path: str = str((self._get_frame_dir(frame) / "colors.yuv").resolve())
        self._threadpool.submit_task(
            write_bytes, out_path=out_path, data=self._yuv_bytes_texture_format(frame)
        )
