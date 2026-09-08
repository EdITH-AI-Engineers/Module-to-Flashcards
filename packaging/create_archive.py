from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


def create_archive(source_dir: Path, destination: Path) -> Path:
    source = Path(source_dir).resolve()
    output = Path(destination).resolve()
    if not source.is_dir():
        raise ValueError(f"portable bundle directory not found: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for candidate in sorted(source.rglob("*")):
            if candidate.is_file():
                relative = candidate.relative_to(source)
                archive.write(candidate, (Path(source.name) / relative).as_posix())
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a ZIP64 archive for the portable Windows bundle."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = create_archive(args.source_dir, args.output)
    print(f"Portable archive created at {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
