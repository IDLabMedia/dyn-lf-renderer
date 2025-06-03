import json

import numpy as np
import jax.numpy as jnp

from pathlib import Path
from typing import List

from rtdlf_preprocessor.video.camera import Camera


def write_bytes(out_path: Path, data: bytes) -> None:
    with open(out_path, "wb") as file:
        file.write(data)


def write_metadata(
    out_dir: Path, total_frames: int, grid_spacing: float, view: np.ndarray
) -> None:
    metadata = {}
    metadata["total_frames"] = total_frames
    metadata["grid_spacing"] = grid_spacing
    metadata["view"] = view.tolist()
    with open(str(out_dir / "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=1)


def write_cameras_info(out_path: Path, cameras: List[Camera]) -> None:
    cam_dict = []
    for camera in cameras:
        json_cam = {}
        json_cam["resolution"] = [camera.resolution.width, camera.resolution.height]
        json_cam["focal"] = [camera.focal.x, camera.focal.y]
        json_cam["pp"] = [camera.pp.x, camera.pp.y]
        json_cam["position"] = [float(x) for x in camera.position]
        json_cam["inv_model"] = np.array(jnp.linalg.inv(camera.model)).tolist()
        json_cam["depth_range"] = [camera.depth_range.near, camera.depth_range.far]
        cam_dict.append(json_cam)

    with open(str(out_path / "cameras.json"), "w") as f:
        json.dump(cam_dict, f, indent=1)
