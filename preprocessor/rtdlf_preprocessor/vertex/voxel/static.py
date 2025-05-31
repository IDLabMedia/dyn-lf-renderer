from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import numpy as np

from rtdlf_preprocessor.vertex.util import intersect2d_np
from rtdlf_preprocessor.writers import write_bytes
#
# def extract_static(vox_frames: List[jnp.ndarray]) -> jnp.ndarray:
#     """Takes n voxel aligned frames, and returns the points that all frames have in common.
#
#     :param vox_frames: The voxel aligned frames, of shape (n,m,3).
#     """
#     static = vox_frames[0]
#     for i in range(1, len(vox_frames)):
#         static = intersect2d(static, vox_frames[i])
#     return static


def extract_static(out_dir: Path, max_frames: int):
    pool = ThreadPoolExecutor(os.cpu_count())
    futures = []
    batch_size_expected = 10
    for frame in range(0, max_frames, batch_size_expected):
        batch_size = (
            batch_size_expected
            if frame + batch_size_expected < max_frames
            else frame + batch_size_expected - max_frames
        )
        futures.append(
            pool.submit(
                _read_batch_vcl,
                out_dir=out_dir,
                batch_start_idx=frame,
                batch_size=batch_size,
            )
        )
    results = [f.result() for f in futures]
    static = results[0]
    for i in range(1, len(results)):
        static = intersect2d_np(static, results[i])

    write_bytes(out_dir / "static.vox", static.tobytes())


def _read_batch_vcl(out_dir: Path, batch_start_idx: int, batch_size: int) -> np.ndarray:
    print(f"batch {batch_start_idx}")
    static = _read_vcl(out_dir / f"frame_{batch_start_idx}/pcl.vox")
    for i in range(batch_start_idx + 1, batch_start_idx + batch_size):
        static = intersect2d_np(static, _read_vcl(out_dir / f"frame_{i}/pcl.vox"))
    return static


def _read_vcl(path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 3)
