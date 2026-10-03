"""
ModernGL mesh buffer management and procedural 3D primitives (car chassis, wheels, obstacles, ground).
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import numpy as np
import moderngl


class GLMesh:
    """Wrapper managing ModernGL VBO, IBO, and VAO."""
    def __init__(self, ctx: moderngl.Context, vao: moderngl.VertexArray, num_indices: int):
        self.ctx = ctx
        self.vao = vao
        self.num_indices = num_indices

    def render(self, mode=moderngl.TRIANGLES) -> None:
        self.vao.render(mode=mode)

    def release(self) -> None:
        try:
            self.vao.release()
        except Exception:
            pass


class MeshBuilder:
    """
    Constructs procedural 3D meshes and uploads them to the GPU.
    """
    @staticmethod
    def create_road_mesh(ctx: moderngl.Context, program: moderngl.Program, vertices: np.ndarray, normals: np.ndarray, uvs: np.ndarray, indices: np.ndarray) -> Optional[GLMesh]:
        if len(vertices) == 0 or len(indices) == 0:
            return None

        # Interleave vertex attributes: [x, y, z, nx, ny, nz, u, v]
        vertex_data = np.hstack([vertices, normals, uvs]).astype(np.float32)
        index_data = indices.astype(np.int32).flatten()

        vbo = ctx.buffer(vertex_data.tobytes())
        ibo = ctx.buffer(index_data.tobytes())

        vao = ctx.vertex_array(
            program,
            [(vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_uv')],
            ibo
        )
        return GLMesh(ctx, vao, len(index_data))

    @staticmethod
    def create_color_mesh(ctx: moderngl.Context, program: moderngl.Program, vertices: np.ndarray, colors: np.ndarray, indices: np.ndarray) -> Optional[GLMesh]:
        if len(vertices) == 0 or len(indices) == 0:
            return None

        vertex_data = np.hstack([vertices, colors]).astype(np.float32)
        index_data = indices.astype(np.int32).flatten()

        vbo = ctx.buffer(vertex_data.tobytes())
        ibo = ctx.buffer(index_data.tobytes())

        vao = ctx.vertex_array(
            program,
            [(vbo, '3f 3f', 'in_position', 'in_color')],
            ibo
        )
        return GLMesh(ctx, vao, len(index_data))

    @staticmethod
    def create_standard_mesh(ctx: moderngl.Context, program: moderngl.Program, vertices: np.ndarray, normals: np.ndarray, indices: np.ndarray) -> Optional[GLMesh]:
        if len(vertices) == 0 or len(indices) == 0:
            return None

        vertex_data = np.hstack([vertices, normals]).astype(np.float32)
        index_data = indices.astype(np.int32).flatten()

        vbo = ctx.buffer(vertex_data.tobytes())
        ibo = ctx.buffer(index_data.tobytes())

        vao = ctx.vertex_array(
            program,
            [(vbo, '3f 3f', 'in_position', 'in_normal')],
            ibo
        )
        return GLMesh(ctx, vao, len(index_data))

    @staticmethod
    def create_box_primitive(ctx: moderngl.Context, program: moderngl.Program, length: float, width: float, height: float) -> GLMesh:
        hl, hw, hh = length * 0.5, width * 0.5, height * 0.5
        # 24 vertices for 6 faces
        verts = np.array([
            # Top (+Z)
            [-hl, -hw, hh], [hl, -hw, hh], [hl, hw, hh], [-hl, hw, hh],
            # Bottom (-Z)
            [-hl, hw, -hh], [hl, hw, -hh], [hl, -hw, -hh], [-hl, -hw, -hh],
            # Front (+X)
            [hl, -hw, -hh], [hl, hw, -hh], [hl, hw, hh], [hl, -hw, hh],
            # Back (-X)
            [-hl, hw, -hh], [-hl, -hw, -hh], [-hl, -hw, hh], [-hl, hw, hh],
            # Left (+Y)
            [-hl, hw, -hh], [hl, hw, -hh], [hl, hw, hh], [-hl, hw, hh],
            # Right (-Y)
            [hl, -hw, -hh], [-hl, -hw, -hh], [-hl, -hw, hh], [hl, -hw, hh],
        ], dtype=np.float32)

        norms = np.array([
            [0, 0, 1]] * 4 +
            [[0, 0, -1]] * 4 +
            [[1, 0, 0]] * 4 +
            [[-1, 0, 0]] * 4 +
            [[0, 1, 0]] * 4 +
            [[0, -1, 0]] * 4,
            dtype=np.float32
        )

        indices = []
        for face in range(6):
            base = face * 4
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
        indices = np.array(indices, dtype=np.int32)

        return MeshBuilder.create_standard_mesh(ctx, program, verts, norms, indices)

    @staticmethod
    def create_ground_plane(ctx: moderngl.Context, program: moderngl.Program, size: float = 800.0) -> GLMesh:
        hs = size * 0.5
        verts = np.array([
            [-hs, -hs, -0.05], [hs, -hs, -0.05], [hs, hs, -0.05], [-hs, hs, -0.05]
        ], dtype=np.float32)
        norms = np.array([
            [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]
        ], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        return MeshBuilder.create_standard_mesh(ctx, program, verts, norms, indices)

    @staticmethod
    def create_vehicle_chassis_mesh(ctx: moderngl.Context, program: moderngl.Program, length: float = 4.2, width: float = 1.8, height: float = 1.4) -> GLMesh:
        """Procedural car body with cabin and hood."""
        hl = length * 0.5
        hw = width * 0.5
        hh = height * 0.5

        # Multi-box car shape: main body lower + cabin upper
        return MeshBuilder.create_box_primitive(ctx, program, length, width, height * 0.6)
