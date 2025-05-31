import sys
import json

import jax.numpy as jnp

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rtdlf_preprocessor.video.camera import (
    Camera,
    DepthRange,
    Focal,
    PrincipalPoint,
    Resolution,
)
from rtdlf_preprocessor.video.color_video import ColorVideo
from rtdlf_preprocessor.video.depth_video import DepthVideo


class DataLoader:
    """Load the input data to python objects"""

    _json_path: Path
    _dataset_path: Path

    def __init__(self, input_dir: Path) -> None:
        self._dataset_path = input_dir
        self._json_path = input_dir / (input_dir.name + ".json")
        if not self._json_path.exists():
            print(
                f"ERROR::DATA_LOADER::JSON::NOT_EXISTING: {self._json_path}",
                file=sys.stderr,
            )
            print(
                f"Please ensure the input folder contains a json file with the same name as the input folder. This file should contain the metadata of the cameras.",
                file=sys.stderr,
            )
            exit(1)

    def _read_json(self) -> Dict:
        data = {}
        with open(self._json_path) as f:
            data = json.load(f)
        return data

    def _parse_defaults(
        self, data: Dict
    ) -> Tuple[
        Optional[Resolution],
        Optional[Focal],
        Optional[PrincipalPoint],
        Optional[DepthRange],
    ]:
        res = data.get("Resolution", None)
        focal = data.get("Focal", None)
        pp = data.get("Depth_range", None)
        depth_range = data.get("Depth_range", None)

        return (
            Resolution(res[0], res[1]) if res else res,
            Focal(focal[0], focal[1]) if focal else focal,
            PrincipalPoint(pp[0], pp[1]) if pp else pp,
            DepthRange(depth_range[0], depth_range[1]) if depth_range else depth_range,
        )

    def _parse_camera(
        self,
        camera_idx: int,
        camera: Dict,
        fb_res: Optional[Resolution],
        fb_focal: Optional[Focal],
        fb_pp: Optional[PrincipalPoint],
        fb_depth_range: Optional[DepthRange],
    ) -> Tuple[
        Resolution,
        Focal,
        PrincipalPoint,
        DepthRange,
    ]:
        raw_res = camera.get("Resolution", None)
        res: Optional[Resolution] = (
            Resolution(raw_res[0], raw_res[1]) if raw_res else fb_res
        )

        raw_focal = camera.get("Focal", None)
        focal: Optional[Focal] = (
            Focal(raw_focal[0], raw_focal[1]) if raw_focal else fb_focal
        )

        raw_pp = camera.get("Depth_range", None)
        pp: Optional[PrincipalPoint] = (
            PrincipalPoint(raw_pp[0], raw_pp[1]) if raw_pp else fb_pp
        )

        raw_depth_range = camera.get("Depth_range", None)
        depth_range: Optional[DepthRange] = (
            DepthRange(raw_depth_range[0], raw_depth_range[1])
            if raw_depth_range
            else fb_depth_range
        )

        if res is None:
            print(
                f"ERROR::DATA_LOADER::CAMERA_{camera_idx}::INVALID_RESOLUTION",
                file=sys.stderr,
            )
            exit(1)
        if focal is None:
            print(
                f"ERROR::DATA_LOADER::CAMERA_{camera_idx}::INVALID_FOCAL",
                file=sys.stderr,
            )
            exit(1)
        if pp is None:
            print(
                f"ERROR::DATA_LOADER::CAMERA_{camera_idx}::INVALID_PRINCIPAL_POINT",
                file=sys.stderr,
            )
            exit(1)
        if depth_range is None:
            print(
                f"ERROR::DATA_LOADER::CAMERA_{camera_idx}::INVALID_DEPTH_RANGE",
                file=sys.stderr,
            )
            exit(1)

        return res, focal, pp, depth_range

    def load_data(self) -> Tuple[List[Camera], List[DepthVideo], List[ColorVideo]]:
        """Load the camera intrinsics and extrinsics, the depth and the color videos.

        :return: The cameras, the depth videos and the color videos
        :rtype: Tuple[List[Camera], List[DepthVideo], List[ColorVideo]]
        """
        data = self._read_json()
        # get fallback data
        fb_res, fb_focal, fb_pp, fb_depth_range = self._parse_defaults(data)

        # enumerate cameras
        cameras: List[Camera] = []
        depth_vids: List[DepthVideo] = []
        color_vids: List[ColorVideo] = []
        for camera_idx, json_cam in enumerate(data["cameras"]):
            # get res, focal, pp and depth_range
            res, focal, pp, depth_range = self._parse_camera(
                camera_idx, json_cam, fb_res, fb_focal, fb_pp, fb_depth_range
            )
            # get model matrix and position
            raw_model = json_cam.get("model", None)
            if raw_model is None:
                print(
                    f"ERROR::DATA_LOADER::CAMERA_{camera_idx}::INVALID_MODEL_MATRIX",
                    file=sys.stderr,
                )
                exit(1)
            model = jnp.array(raw_model)
            if model.shape != (4, 4):
                print(
                    f"ERROR::DATA_LOADER::CAMERA_{camera_idx}::INVALID_MODEL_MATRIX_SHAPE: {model.shape}, but excpected (4,4)",
                    file=sys.stderr,
                )
                exit(1)
            position = model[:3, -1]

            # add the camera
            cameras.append(
                Camera(
                    res,
                    focal,
                    pp,
                    depth_range,
                    model,
                    position,
                    camera_idx,
                )
            )

            # add depth video
            depth_vids.append(
                DepthVideo(
                    str((self._dataset_path / json_cam["NameDepth"]).resolve()).replace(
                        "mp4", "yuv"
                    ),
                    res.width,
                    res.height,
                )
            )

            # add color video
            color_vids.append(
                ColorVideo(
                    str((self._dataset_path / json_cam["NameColor"]).resolve()),
                    res.width,
                    res.height,
                )
            )

        return cameras, depth_vids, color_vids
