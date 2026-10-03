"""
Deterministic episode replay player with frame seeking and playback controls.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional


class EpisodeReplayPlayer:
    """
    Plays back recorded episode frames with play/pause, seek, and scrub controls.
    """
    def __init__(self):
        self.metadata: Dict[str, Any] = {}
        self.frames: List[Dict[str, Any]] = []
        self.current_frame_idx: int = 0
        self.is_playing: bool = False
        self.playback_speed: float = 1.0

    def load_recording(self, recording_data: Dict[str, Any]) -> None:
        self.metadata = recording_data.get('metadata', {})
        self.frames = recording_data.get('frames', [])
        self.current_frame_idx = 0
        self.is_playing = False

    @property
    def total_frames(self) -> int:
        return len(self.frames)

    def play(self) -> None:
        self.is_playing = True

    def pause(self) -> None:
        self.is_playing = False

    def toggle_play(self) -> bool:
        self.is_playing = not self.is_playing
        return self.is_playing

    def seek(self, frame_idx: int) -> None:
        if self.frames:
            self.current_frame_idx = max(0, min(frame_idx, len(self.frames) - 1))

    def step_forward(self) -> Optional[Dict[str, Any]]:
        if not self.frames:
            return None
        if self.current_frame_idx < len(self.frames) - 1:
            self.current_frame_idx += 1
        else:
            self.is_playing = False
        return self.get_current_frame()

    def step_backward(self) -> Optional[Dict[str, Any]]:
        if not self.frames:
            return None
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
        return self.get_current_frame()

    def get_current_frame(self) -> Optional[Dict[str, Any]]:
        if not self.frames or self.current_frame_idx >= len(self.frames):
            return None
        return self.frames[self.current_frame_idx]
