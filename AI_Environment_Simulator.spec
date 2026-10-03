# -*- mode: python ; coding: utf-8 -*-


import os

# Resolve entry point relative to this spec file — never hardcode a
# development-machine path.
_spec_dir = os.path.dirname(os.path.abspath(
    SPECPATH if 'SPECPATH' in globals() else __file__))

a = Analysis(
    [os.path.join(_spec_dir, 'main.py')],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['moderngl', 'glcontext', 'OpenGL', 'OpenGL.GL', 'pygame', 'numpy', 'cv2', 'PIL', 'shapely', 'scipy', 'gymnasium', 'sim_core', 'sim_env', 'sim_net', 'sim_client', 'sim_ui', 'sim_render', 'sim_recorder', 'sim_project'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'tensorflow', 'tensorboard', 'pandas', 'pyarrow', 'botocore', 'boto3', 'openpyxl', 'weasyprint', 'pdf2image', 'pypdf', 'PyPDF2', 'matplotlib', 'IPython', 'notebook', 'jupyter', 'sympy', 'transformers', 'ultralytics', 'unsloth', 'peft', 'scikit_learn', 'sklearn', 'seaborn', 'pytest'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AI_Environment_Simulator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AI_Environment_Simulator',
)
