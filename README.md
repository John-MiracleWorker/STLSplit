# STLSplit

CLI-first Python library for splitting STL/OBJ/3MF meshes into printer-sized pieces with connector support.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# Optional but recommended for robust booleans
pip install "manifold3d>=2.3"

stlsplit path/to/helmet.stl --output output/helmet.3mf
```

## Notes

- Units are assumed to be millimeters.
- 3MF export uses `trimesh` Scene export; if it fails, install a newer `trimesh`.
- Boolean operations require a backend (default: `manifold3d`).

## Config

Defaults live in `config.yaml`. Override with CLI flags or `--config`.
