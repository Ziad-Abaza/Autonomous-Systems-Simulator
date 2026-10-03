"""
GLSL vertex and fragment shaders for modern programmable OpenGL rendering.
"""

from __future__ import annotations

# Standard lit mesh shader (vehicle, barriers, ground)
STANDARD_VS = """
#version 330 core
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;

uniform mat4 u_mvp;
uniform mat4 u_model;

out vec3 v_normal;
out vec3 v_world_pos;

void main() {
    v_world_pos = vec3(u_model * vec4(in_position, 1.0));
    v_normal = mat3(transpose(inverse(u_model))) * in_normal;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

STANDARD_FS = """
#version 330 core
in vec3 v_normal;
in vec3 v_world_pos;

uniform vec3 u_color;
uniform vec3 u_light_dir;
uniform float u_ambient;

out vec4 fragColor;

void main() {
    vec3 norm = normalize(v_normal);
    vec3 lightDir = normalize(u_light_dir);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(0.85, 0.85, 0.85);
    vec3 ambient = u_ambient * vec3(0.35, 0.35, 0.4);
    vec3 result = (ambient + diffuse) * u_color;
    fragColor = vec4(result, 1.0);
}
"""

# Road surface shader with procedural dashed center line and solid white edge lines
ROAD_VS = """
#version 330 core
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;

uniform mat4 u_mvp;

out vec2 v_uv;
out vec3 v_normal;

void main() {
    v_uv = in_uv;
    v_normal = in_normal;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

ROAD_FS = """
#version 330 core
in vec2 v_uv;
in vec3 v_normal;

uniform vec3 u_light_dir;
uniform float u_ambient;

out vec4 fragColor;

void main() {
    // Base asphalt dark gray
    vec3 asphalt = vec3(0.18, 0.18, 0.20);

    // Lateral position: v_uv.x in [0.0, 1.0]
    // 0.0 is left edge, 0.5 is centerline, 1.0 is right edge
    float lat = v_uv.x;
    float long_s = v_uv.y;

    // Centerline: dashed white line
    bool is_center = abs(lat - 0.5) < 0.015;
    bool is_dash = fract(long_s * 0.3) < 0.55;

    // Edge lines: solid white lines
    bool is_left_edge = abs(lat - 0.03) < 0.012;
    bool is_right_edge = abs(lat - 0.97) < 0.012;

    vec3 color = asphalt;
    if (is_left_edge || is_right_edge) {
        color = vec3(0.92, 0.92, 0.92);
    } else if (is_center && is_dash) {
        color = vec3(0.95, 0.95, 0.95);
    }

    vec3 norm = normalize(v_normal);
    vec3 lightDir = normalize(u_light_dir);
    float diff = max(dot(norm, lightDir), 0.0) * 0.7;
    vec3 result = (u_ambient * 0.4 + diff) * color;

    fragColor = vec4(result, 1.0);
}
"""

# Vertex color shader (curbs, debug elements)
COLOR_VS = """
#version 330 core
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_color;

uniform mat4 u_mvp;
out vec3 v_color;

void main() {
    v_color = in_color;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

COLOR_FS = """
#version 330 core
in vec3 v_color;
out vec4 fragColor;

void main() {
    fragColor = vec4(v_color, 1.0);
}
"""

# Line shader (LiDAR rays, trajectories, collision wireframes)
LINE_VS = """
#version 330 core
layout (location = 0) in vec3 in_position;

uniform mat4 u_mvp;

void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

LINE_FS = """
#version 330 core
uniform vec4 u_color;
out vec4 fragColor;

void main() {
    fragColor = u_color;
}
"""
