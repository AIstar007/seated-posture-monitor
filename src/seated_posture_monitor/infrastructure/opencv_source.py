"""OpenCV adapter implementing the application's FrameSource port."""
from __future__ import annotations

from typing import Any

import cv2


class OpenCVVideoSource:
    def __init__(self, source: Any, width: int | None = None, height: int | None = None) -> None:
        self._source = source
        self._capture = cv2.VideoCapture(source)
        self._is_camera = isinstance(source, int)
        # Camera backends may ignore these requests; callers must inspect the
        # actual resolution after opening rather than assume it was honoured.
        if self._is_camera:
            if width is not None:
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            if height is not None:
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    @property
    def is_camera(self) -> bool:
        return self._is_camera

    def is_opened(self) -> bool:
        return bool(self._capture.isOpened())

    def read(self) -> tuple[bool, Any]:
        try:
            return self._capture.read()
        except cv2.error:
            return False, None

    def fps(self) -> float:
        return float(self._capture.get(cv2.CAP_PROP_FPS))

    def frame_count(self) -> float:
        return float(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def position_seconds(self) -> float:
        return float(self._capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0

    def position_frames(self) -> float:
        return float(self._capture.get(cv2.CAP_PROP_POS_FRAMES))

    def seek_frame(self, frame_number: int) -> bool:
        return bool(self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number))

    def release(self) -> None:
        self._capture.release()
