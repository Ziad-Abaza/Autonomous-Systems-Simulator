"""
High-performance Offscreen Framebuffer (FBO) rendering for synthetic onboard vehicle camera sensor.
Supports direct read_into preallocated NumPy buffers and double-buffered PBO transfers.
Preserves (H, W, 3) uint8 RGB layout and bottom-to-top orientation correction.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import moderngl


class OffscreenFBO:
    """
    Renders 3D scene to an offscreen texture and transfers RGB frames to CPU.
    """
    def __init__(self, ctx: moderngl.Context, width: int = 84, height: int = 84, use_pbo: bool = False):
        self.ctx = ctx
        self.width = int(width)
        self.height = int(height)
        self.use_pbo = use_pbo

        # Offscreen Texture & Depth Buffer
        self.color_texture = ctx.texture((self.width, self.height), 4)
        self.depth_renderbuffer = ctx.depth_renderbuffer((self.width, self.height))
        self.fbo = ctx.framebuffer(
            color_attachments=[self.color_texture],
            depth_attachment=self.depth_renderbuffer
        )

        # Preallocated CPU memory buffers to eliminate per-step garbage collection overhead
        self._buf_size = self.height * self.width * 3
        self._raw_buffer = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self._out_buffer = np.empty((self.height, self.width, 3), dtype=np.uint8)

        # PBO Double Buffering (Ping-Pong) for asynchronous readback
        self.pbo_ping: Optional[moderngl.Buffer] = None
        self.pbo_pong: Optional[moderngl.Buffer] = None
        self._pbo_flip = 0

        if self.use_pbo:
            try:
                self.pbo_ping = ctx.buffer(reserve=self._buf_size)
                self.pbo_pong = ctx.buffer(reserve=self._buf_size)
            except Exception:
                self.use_pbo = False

    def bind(self) -> None:
        """Binds FBO as active render target and clears buffers."""
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.fbo.clear(0.53, 0.81, 0.92, 1.0)  # Sky blue background

    def read_rgb(self) -> np.ndarray:
        """
        Fast GPU -> CPU readback into preallocated buffer.
        Returns contiguous (H, W, 3) uint8 RGB array with vertical flip corrected.
        """
        if self.use_pbo and self.pbo_ping and self.pbo_pong:
            return self.read_rgb_pbo()

        # Direct read_into avoids allocating Python bytes on every step
        self.fbo.read_into(self._raw_buffer, components=3, alignment=1)
        # Flip vertically to convert OpenGL bottom-up coordinates to standard image coordinates
        return np.ascontiguousarray(np.flipud(self._raw_buffer))

    def read_rgb_pbo(self) -> np.ndarray:
        """
        Asynchronous Pixel Buffer Object transfer.
        Writes current frame to current PBO while reading previous frame into NumPy array.
        """
        curr_pbo = self.pbo_ping if self._pbo_flip == 0 else self.pbo_pong
        prev_pbo = self.pbo_pong if self._pbo_flip == 0 else self.pbo_ping

        # Asynchronously schedule readback from FBO into GPU PBO
        self.fbo.read_into(curr_pbo, components=3, alignment=1)

        # Read back from the other PBO (which finished rendering in the previous step)
        prev_pbo.read_into(self._raw_buffer)
        self._pbo_flip = 1 - self._pbo_flip

        return np.ascontiguousarray(np.flipud(self._raw_buffer))

    def read_rgb_legacy(self) -> np.ndarray:
        """Original unoptimized readback allocating new bytes each frame for benchmarking."""
        raw = self.fbo.read(components=3, alignment=1)
        img = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        return np.ascontiguousarray(np.flipud(img))

    def release(self) -> None:
        try:
            if self.pbo_ping: self.pbo_ping.release()
            if self.pbo_pong: self.pbo_pong.release()
            self.color_texture.release()
            self.depth_renderbuffer.release()
            self.fbo.release()
        except Exception:
            pass
