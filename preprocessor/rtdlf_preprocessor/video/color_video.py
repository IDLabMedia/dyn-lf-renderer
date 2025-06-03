from typing import Tuple
import cv2

from cv2.typing import MatLike
import numpy as np
import jax.numpy as jnp

from rtdlf_preprocessor.video.video import Video


class ColorVideo(Video):
    # yuv to rgb transformation matrix
    _mat_sycc_yuv_to_rgb = np.array(
        [
            [1.0, 0.000037, 1.401988],
            [1.0, -0.344113, -0.714104],
            [1.0, 1.771978, 0.000135],
        ]
    )
    _mat_sycc_rgb_to_yuv = np.linalg.inv(_mat_sycc_yuv_to_rgb)
    _M = 255
    _Z = 128

    def __init__(self, path: str, width: int, height: int) -> None:
        super().__init__(path, width, height)

    def _get_frame_data(self, frame: int) -> MatLike:
        vid = cv2.VideoCapture(self.path)
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
        read, frame_data = vid.read()
        if not read:
            raise IOError(f"Failed to read frame {frame} of file: {self.path}")
        return frame_data

    def _sycc_yuv_to_rgb(self, image: np.ndarray) -> np.ndarray:
        yuv = image.astype(np.float32)
        # y' = y / M
        # u' = (u - Z)/M
        # v' = (u - Z)/M
        yuv[:, :, 1:] -= self._Z
        yuv /= self._M

        rgb = yuv @ self._mat_sycc_yuv_to_rgb.T  # Matrix multiplication per pixel

        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = (rgb * self._M).round().astype(np.uint8)

        return rgb

    def _sycc_rgb_to_yuv(self, image: np.ndarray) -> np.ndarray:
        rgb = image.astype(np.float32) / self._M

        yuv = rgb @ self._mat_sycc_rgb_to_yuv.T

        yuv *= self._M
        yuv[:, :, 1:] += self._Z

        yuv = np.clip(yuv, 0.0, self._M)
        yuv = yuv.round().astype(np.uint8)

        return yuv

    def get_frame(self, frame: int) -> jnp.ndarray:
        return jnp.array(self._get_frame_data(frame))

    def get_colors(self, frame: int) -> np.ndarray:
        return np.array(self._get_frame_data(frame))

    def total_frames(self) -> int:
        vid = cv2.VideoCapture(self.path)
        return int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_yuv_interleaved_frame_np(self, frame: int) -> np.ndarray:
        rgb = cv2.cvtColor(self._get_frame_data(frame), cv2.COLOR_BGR2RGB)
        return self._sycc_rgb_to_yuv(rgb)

    def get_yuv_interleaved_frame(self, frame: int) -> jnp.ndarray:
        return jnp.array(self.get_yuv_interleaved_frame_np(frame))

    def get_yuv420_frame(self, frame: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        yuv_interleaved = self.get_yuv_interleaved_frame_np(frame)

        y = yuv_interleaved[:, :, 0]
        u = yuv_interleaved[:, :, 1]
        v = yuv_interleaved[:, :, 2]

        u_420 = cv2.resize(
            u, (u.shape[1] // 2, u.shape[0] // 2), interpolation=cv2.INTER_LINEAR
        )
        v_420 = cv2.resize(
            v, (v.shape[1] // 2, v.shape[0] // 2), interpolation=cv2.INTER_LINEAR
        )

        return y, u_420, v_420

    def get_flat_yuv420_frame(self, frame: int) -> np.ndarray:
        y, u, v = self.get_yuv420_frame(frame)
        data = np.append(y, u)
        data = np.append(data, v)
        return data

    def write_yuv420_frame(self, path: str, frame: int):
        y, u, v = self.get_yuv420_frame(frame)

        with open(path, "wb") as f:
            f.write(y.tobytes())
            f.write(u.tobytes())
            f.write(v.tobytes())

    def save_frame(self, path: str, frame: int) -> None:
        pixels = self.get_colors(frame)
        cv2.imwrite(path, pixels)
