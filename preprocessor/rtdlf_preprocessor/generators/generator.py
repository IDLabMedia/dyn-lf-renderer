from pathlib import Path
from typing import List

from rtdlf_preprocessor.video.camera import Camera
from rtdlf_preprocessor.threadpool import ThreadPool


class Generator:
    _out_dir: Path  # directory to output to
    _cameras: List[Camera]  # camera metadata
    _max_frames: int  # max amount of frames per video
    _threadpool: ThreadPool

    def __init__(self, out_dir: Path, cameras: List[Camera], max_frames: int) -> None:
        self._out_dir = out_dir
        self._cameras = cameras
        self._max_frames = max_frames
        self._threadpool = ThreadPool()

    def _get_frame_dir(self, frame: int) -> Path:
        frame_dir = self._out_dir / f"frame_{frame}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        return frame_dir
