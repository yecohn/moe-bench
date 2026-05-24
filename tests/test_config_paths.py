"""Pin: every backend `python:` path in checked-in configs uses the
canonical `.backends/<name>/bin/python` location, never an absolute
host path. Lets the same configs work on the host (with .backends/
populated) and inside the Docker image."""

from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _iter_backend_python_paths():
    for config_path in sorted(CONFIGS_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(config_path.read_text()) or {}
        backends = cfg.get("backends") or {}
        for backend_name, backend_cfg in backends.items():
            if not isinstance(backend_cfg, dict):
                continue
            py = backend_cfg.get("python")
            if py is None:
                continue
            yield config_path.name, backend_name, py


def test_all_backend_python_paths_use_canonical_dot_backends_location():
    for config_name, backend_name, py in _iter_backend_python_paths():
        assert py == f".backends/{backend_name}/bin/python", (
            f"{config_name} has non-canonical python path for {backend_name}: {py!r}"
        )
