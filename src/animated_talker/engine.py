"""Abstract base class for talking-head animation engines."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TalkingHeadEngine(ABC):
    """Base class for talking-head animation engines.

    Each engine takes a portrait image and audio waveform, and produces
    animated video frames with lip-sync and (optionally) head motion.
    """

    @abstractmethod
    def generate_frames(
        self,
        image: np.ndarray,
        audio: np.ndarray,
        fps: int = 25,
        **kwargs,
    ) -> list[np.ndarray]:
        """Generate animated talking-head frames.

        Args:
            image: Full-resolution BGR image of the source portrait.
            audio: Float32 mono waveform at 16 kHz.
            fps: Target video frame rate.
            **kwargs: Engine-specific parameters.

        Returns:
            List of BGR frames (all same resolution).
        """
        ...
