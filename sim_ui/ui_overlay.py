"""
2D UI Overlay renderer using Pygame Surface rasterization mapped onto a ModernGL blended texture.
Provides technical simulation HUD, telemetry widgets, reward breakdown bars,
track editor controls, and camera Picture-in-Picture.
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional, Tuple
import pygame
import numpy as np
import moderngl

# Simple overlay shader
OVERLAY_VS = """
#version 330 core
layout (location = 0) in vec2 in_pos;
layout (location = 1) in vec2 in_uv;

out vec2 v_uv;

void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

OVERLAY_FS = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_texture;
out vec4 fragColor;

void main() {
    fragColor = texture(u_texture, v_uv);
}
"""


class UIOverlayRenderer:
    """
    Manages 2D UI rasterization and ModernGL texture blitting.
    """
    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height

        self.ui_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.ui_texture = ctx.texture((width, height), 4)
        self.ui_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Fullscreen quad
        quad_verts = np.array([
            # pos(x, y), uv(u, v)
            [-1.0,  1.0,  0.0, 0.0],
            [-1.0, -1.0,  0.0, 1.0],
            [ 1.0,  1.0,  1.0, 0.0],
            [ 1.0, -1.0,  1.0, 1.0],
        ], dtype=np.float32)

        self.vbo = ctx.buffer(quad_verts.tobytes())
        self.prog = ctx.program(vertex_shader=OVERLAY_VS, fragment_shader=OVERLAY_FS)
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, '2f 2f', 'in_pos', 'in_uv')])

        # Font
        pygame.font.init()
        self.font_small = pygame.font.SysFont("Segoe UI", 13)
        self.font_bold = pygame.font.SysFont("Segoe UI", 14, bold=True)
        self.font_title = pygame.font.SysFont("Segoe UI", 16, bold=True)
        self.font_mono = pygame.font.SysFont("Consolas", 13)

    def resize(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.ui_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.ui_texture.release()
        self.ui_texture = self.ctx.texture((width, height), 4)
        self.ui_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

    def clear(self) -> None:
        self.ui_surface.fill((0, 0, 0, 0))

    def render_to_screen(self) -> None:
        """Uploads UI surface pixels to texture and draws blended fullscreen quad."""
        # Convert Pygame surface to raw RGBA bytes
        raw_data = pygame.image.tobytes(self.ui_surface, 'RGBA', False)
        self.ui_texture.write(raw_data)

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.ui_texture.use(location=0)
        self.prog['u_texture'].value = 0
        self.vao.render(mode=moderngl.TRIANGLE_STRIP)

        self.ctx.enable(moderngl.DEPTH_TEST)
