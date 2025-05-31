import jax.numpy as jnp

from enum import Enum
from pathlib import Path
from typing import List

from rtdlf_preprocessor.vertex.point_cloud import (
    point_cloud,
    voxel_color_cloud,
)
from rtdlf_preprocessor.vertex.util import sort_indices
from rtdlf_preprocessor.util import write_progress
from rtdlf_preprocessor.video.color_video import ColorVideo
from .generator import Generator
from rtdlf_preprocessor.writers import write_bytes
from rtdlf_preprocessor.video.camera import Camera
from rtdlf_preprocessor.video.depth_video import DepthVideo
from rtdlf_preprocessor.vertex.depth_processor import get_cameras_centers


class VertexFormat(Enum):
    RAW_POINT_CLOUD = "RAW_POINT_CLOUD"
    VOXEL_CLOUD = "VOXEL_CLOUD"
    SUBSAMPLED_DEPTH = "SUBSAMPLED_DEPTH"
    CENTERS = "CENTERS"


class VertexGenerator(Generator):
    _depth_videos: List[DepthVideo]

    def __init__(
        self,
        out_dir: Path,
        cameras: List[Camera],
        max_frames: int,
        depth_videos: List[DepthVideo],
        color_videos: List[ColorVideo],
        grid_spacing: float,
    ) -> None:
        super().__init__(out_dir, cameras, max_frames)
        self._depth_videos = depth_videos
        self._color_videos = color_videos
        self._grid_spacing = grid_spacing

    def generate(self, format: VertexFormat, *args, **kwargs) -> None:
        """Generate the requested vertex format and save it to file"""
        match format:
            case VertexFormat.RAW_POINT_CLOUD:
                self._generate_raw_point_cloud()
            case VertexFormat.VOXEL_CLOUD:
                self._generate_voxel_cloud()
            case VertexFormat.SUBSAMPLED_DEPTH:
                self._generate_subsampled_depth_frames()
            case VertexFormat.CENTERS:
                self._generate_centers()
        print(" Done")

    def _generate_centers(self) -> None:
        centers = get_cameras_centers(
            self._cameras, self._depth_videos, self._max_frames
        )
        out_path = (self._out_dir / "centers.bin").resolve()
        write_bytes(out_path, centers.tobytes())

    def _generate_subsampled_depth_frames(self) -> None:
        for frame in range(self._max_frames):
            write_progress(frame + 1, self._max_frames, "Generating subsampled depth")
            out_path: str = str((self._get_frame_dir(frame) / "depth.raw").resolve())
            data = []
            for vid in self._depth_videos:
                data.append(vid.get_depth_frame(frame))
            data = jnp.concatenate(data).ravel()
            write_bytes(Path(out_path), jnp.array(data).tobytes())

    def _generate_raw_point_cloud(self) -> None:
        for frame in range(self._max_frames):
            write_progress(frame + 1, self._max_frames, "Generating raw point cloud")
            out_path: str = str((self._get_frame_dir(frame) / "pcl.raw").resolve())
            self._threadpool.submit_task(
                write_bytes,
                out_path=out_path,
                data=point_cloud(self._cameras, self._depth_videos, frame).tobytes(),
            )

    def _generate_voxel_cloud(self) -> None:
        for frame in range(self._max_frames):
            write_progress(
                frame + 1, self._max_frames, "Generating voxel aligned point cloud"
            )

            vcl_out: Path = (self._get_frame_dir(frame) / "pcl.vox").resolve()
            colors_out: Path = (
                self._get_frame_dir(frame) / "fallback_colors.rgb"
            ).resolve()

            vcl, colors = voxel_color_cloud(
                self._cameras,
                self._depth_videos,
                self._color_videos,
                frame,
                self._grid_spacing,
            )
            indices = sort_indices(vcl)
            write_bytes(vcl_out, vcl[indices].tobytes())
            write_bytes(colors_out, colors[indices].tobytes())

        # extract_static(self._out_dir, self._max_frames)
