"""
Automated Windows Standalone Executable Packaging Script using PyInstaller.
Packages the entire 3D AI Environment Simulator into a standalone distribution
that requires NO Python, NO Docker, and NO developer tools on the target system.
Excludes extraneous site-packages to produce a lean, fast, robust application bundle.
"""

from __future__ import annotations
import os
import sys
import shutil
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def build_standalone_executable():
    print("=" * 65)
    print("BUILDING STANDALONE WINDOWS APPLICATION PACKAGE")
    print("=" * 65)

    dist_dir = os.path.join(PROJECT_ROOT, "dist")
    build_temp_dir = os.path.join(PROJECT_ROOT, "build_temp")
    main_py = os.path.join(PROJECT_ROOT, "main.py")

    # Essential modules for the simulation platform
    hidden_imports = [
        "moderngl",
        "glcontext",
        "OpenGL",
        "OpenGL.GL",
        "pygame",
        "numpy",
        "cv2",
        "PIL",
        "shapely",
        "scipy",
        "gymnasium",
        "sim_core",
        "sim_env",
        "sim_net",
        "sim_client",
        "sim_ui",
        "sim_render",
        "sim_recorder",
        "sim_project",
    ]

    # Exclude massive unrelated data-science / cloud libraries from the standalone app bundle
    excludes = [
        "torch",
        "torchvision",
        "tensorflow",
        "tensorboard",
        "pandas",
        "pyarrow",
        "botocore",
        "boto3",
        "openpyxl",
        "weasyprint",
        "pdf2image",
        "pypdf",
        "PyPDF2",
        "matplotlib",
        "IPython",
        "notebook",
        "jupyter",
        "sympy",
        "transformers",
        "ultralytics",
        "unsloth",
        "peft",
        "scikit_learn",
        "sklearn",
        "seaborn",
        "pytest",
    ]

    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--name=AI_Environment_Simulator",
        "--noconfirm",
        "--onedir",
        "--windowed",
        f"--distpath={dist_dir}",
        f"--workpath={build_temp_dir}",
    ]

    for hi in hidden_imports:
        cmd.extend(["--hidden-import", hi])

    for ex in excludes:
        cmd.extend(["--exclude-module", ex])

    cmd.append(main_py)

    print(f"[Build] Launching optimized PyInstaller build...")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print("[Build] ERROR: PyInstaller packaging failed!")
        return False

    # Copy preset files into the packaged directory
    output_dir = os.path.join(dist_dir, "AI_Environment_Simulator")
    presets_dest = os.path.join(output_dir, "presets")

    from sim_project.presets import save_default_presets
    save_default_presets(presets_dest)

    # Copy README
    readme_src = os.path.join(PROJECT_ROOT, "README.md")
    if os.path.exists(readme_src):
        shutil.copy(readme_src, output_dir)

    print("=" * 65)
    print(f"SUCCESS: Standalone Windows application built successfully!")
    print(f"Executable location: {os.path.join(output_dir, 'AI_Environment_Simulator.exe')}")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = build_standalone_executable()
    sys.exit(0 if success else 1)
