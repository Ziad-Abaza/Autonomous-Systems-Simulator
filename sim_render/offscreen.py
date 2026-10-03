"""
Offscreen Framebuffer (FBO) rendering for synthetic onboard vehicle camera sensor.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import moderngl


class OffscreenFBO:
    """
    Renders offscreen to a texture and extracts RGB frames.
    """
    def __init__(self, ctx: moderngl.Context, width: int = 84, height: int = 84):
        self.ctx = ctx
        self.width = width
        self.height = height

        self.color_texture = ctx.texture((width, height), 4)
        self.depth_renderbuffer = ctx.depth_renderbuffer((width, height))
        self.fbo = ctx.framebuffer(
            color_attachments=[self.color_texture],
            depth_attachment=self.depth_renderbuffer
        )

    def bind(self) -> None:
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.fbo.clear(0.53, 0.81, 0.92, 1.0)  # Sky blue background

    def read_rgb(self) -> np.ndarray:
        """Reads back pixels into uint8 (H, W, 3) RGB array."""
        raw = self.fbo.read(components=3, alignment=1)
        img = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        # OpenGL images are bottom-up, flip vertically
        return np.ascontiguousarray(np.flipud(img))

    def release(self) -> None:
        try:
            self.color_texture.release()
            self.depth_renderbuffer.release()
            self.fbo.release()
        except Exception:
            pass
