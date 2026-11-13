"""Export trusted tensor weights under pinned CPU Torch, rejecting pickle fallback."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def validate_loader(source):
    tree = ast.parse(source, feature_version=(3, 10))
    allowed_imports = {"base64", "io", "torch", "zlib"}
    jit_load = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name not in allowed_imports for alias in node.names
        ):
            raise ValueError("Export loader has a training/pickle dependency")
        if isinstance(node, ast.ImportFrom):
            raise ValueError("Export loader must use only vetted stdlib/Torch imports")  # noqa: TRY004
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if (
                name == "torch.load"
                or "pickle" in name
                or name in ("open", "eval", "exec", "__import__")
            ):
                raise ValueError(
                    "Export loader must not use external files or pickle fallback"
                )
            jit_load |= name == "torch.jit.load"
    if not jit_load:
        raise ValueError("Export is not the self-contained TorchScript loader")


def export_model(checkpoint: Path, output: Path) -> dict:
    import torch
    import torch.version

    from tools.train_vision_v3 import CellCNN
    from utils import generate_torch_loader_snippet

    if not torch.__version__.startswith("2.7.0") or torch.version.cuda is not None:
        raise ValueError("Export requires the pinned CPU Torch 2.7.0 environment")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = CellCNN().eval()
    model.load_state_dict(weights["state_dict"], strict=True)
    example = torch.zeros(1, 3, 48, 48)
    loader = generate_torch_loader_snippet(
        model, example_inputs=example, prefer="trace", compression="zlib"
    )
    validate_loader(loader)
    namespace = {}
    began = perf_counter()
    exec(loader, namespace)  # noqa: S102 -- vetted, locally generated TorchScript loader
    restored = namespace["get_model"]()
    load_seconds = perf_counter() - began
    errors = []
    generator = torch.Generator().manual_seed(20261109)
    with torch.inference_mode():
        for n in (1, 7, 225):
            crops = torch.rand(n, 3, 48, 48, generator=generator)
            expected, actual = model(crops), restored(crops)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            if not torch.equal(actual.argmax(1), expected.argmax(1)):
                raise ValueError("Export top-one mismatch")
            errors.append(float((actual - expected).abs().max()))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "model_loader.py").write_text(loader)
    torch.jit.save(restored, output / "model.pt")
    report = {
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "export_sha256": hashlib.sha256((output / "model.pt").read_bytes()).hexdigest(),
        "loader_sha256": hashlib.sha256(loader.encode()).hexdigest(),
        "export_bytes": (output / "model.pt").stat().st_size,
        "loader_bytes": len(loader.encode()),
        "parity_passed": True,
        "batches": [1, 7, 225],
        "max_logit_errors": errors,
        "warm_load_seconds": load_seconds,
        "temperature": weights.get("temperature", 1.0),
        "torch_version": torch.__version__,
    }
    code = """import json,time
t=time.perf_counter()
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
u=time.perf_counter()
import runpy
m=runpy.run_path(__import__('sys').argv[1])['get_model']()
v=time.perf_counter()
with torch.inference_mode():m(torch.zeros(225,3,48,48))
print(json.dumps({'cold_process_seconds':time.perf_counter()-t,'torch_import_seconds':u-t,'cold_load_seconds':v-u}))
"""
    cold = subprocess.run(
        [sys.executable, "-c", code, str(output / "model_loader.py")],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    report.update(json.loads(cold.stdout))
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
