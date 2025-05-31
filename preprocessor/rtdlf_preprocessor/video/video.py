import os


class Video:
    path: str
    width: int
    height: int

    def __init__(self, path: str, width: int, height: int) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Failed to find video: {path}.")
        self.path = path
        self.height = height
        self.width = width
