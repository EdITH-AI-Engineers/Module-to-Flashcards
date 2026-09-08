from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules


PROJECT_ROOT = Path(SPECPATH).resolve().parent.parent
hiddenimports = (
    collect_submodules("uvicorn")
    + collect_submodules("transformers")
    + collect_submodules("sentencepiece")
)
binaries = collect_dynamic_libs("torch") + collect_dynamic_libs("llama_cpp")

a = Analysis(
    [str(PROJECT_ROOT / "portable_launcher.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ModuleToFlashcards",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory="runtime",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ModuleToFlashcards",
)
