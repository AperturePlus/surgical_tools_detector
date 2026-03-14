#!/usr/bin/env python3
"""Export a YOLOv8/YOLO11 .pt model to ONNX for different ONNX Runtime
inference backends (CPU, CUDA, DirectML / AMD Radeon).

Usage
-----
  # CPU or DirectML/Radeon (fp32 ONNX):
  python scripts/export_model.py --model assets/v8s.pt --backend cpu
  python scripts/export_model.py --model assets/v8s.pt --backend directml

  # CUDA (fp16 ONNX — smaller model, faster GPU inference):
  python scripts/export_model.py --model assets/v8s.pt --backend cuda

  # Export all variants at once (produces best.onnx and best_fp16.onnx):
  python scripts/export_model.py --model assets/v8s.pt --backend all

Output files
------------
  assets/best.onnx        — fp32 ONNX (used by cpu and directml builds)
  assets/best_fp16.onnx   — fp16 ONNX (used by cuda build)

Notes
-----
  * All exports use opset 17 and graph simplification (onnxsim).
  * The fp16 model requires ONNX Runtime >= 1.16 with a CUDA-capable GPU
    (ORT CUDA EP handles fp16 input/output natively).  The bundled C++
    detector reads float32; if you switch to the fp16 model you must also
    update YoloOnnxDetector to use Ort::Value::CreateTensor<Ort::Float16_t>.
  * For local development run this script once before invoking cmake.
"""

import argparse
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _export_onnx(model_path: Path, output_path: Path, half: bool,
                 imgsz: int, opset: int, simplify: bool) -> None:
    """Export *model_path* to ONNX and move the result to *output_path*."""
    try:
        from ultralytics import YOLO  # type: ignore[import]
    except ImportError:
        print(
            "ERROR: 'ultralytics' is not installed.\n"
            "Install it with:  pip install ultralytics onnx onnxsim",
            file=sys.stderr,
        )
        sys.exit(1)

    model = YOLO(str(model_path))

    exported = model.export(
        format="onnx",
        opset=opset,
        simplify=simplify,
        half=half,
        dynamic=False,
        imgsz=imgsz,
    )

    # ultralytics returns the path of the exported file (str or Path).
    # Fall back to the default ultralytics naming convention if needed.
    src = Path(exported) if exported else model_path.with_suffix(".onnx")

    if not src.exists():
        print(f"ERROR: exported file not found at '{src}'", file=sys.stderr)
        sys.exit(1)

    if src.resolve() != output_path.resolve():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(output_path))

    print(f"[export] {'fp16' if half else 'fp32'} ONNX -> {output_path}")


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------

def export_for_backend(model_path: Path, backend: str,
                       output_dir: Path, imgsz: int,
                       opset: int, simplify: bool) -> None:
    """Export *model_path* for the requested *backend*."""

    output_dir.mkdir(parents=True, exist_ok=True)

    if backend in ("cpu", "directml"):
        # Standard fp32 ONNX — works with CPU EP and DirectML EP.
        _export_onnx(
            model_path,
            output_path=output_dir / "best.onnx",
            half=False,
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
        )

    elif backend == "cuda":
        # fp16 ONNX — smaller and faster on CUDA-capable GPUs.
        # NOTE: the bundled C++ detector currently uses float32 tensors;
        # update YoloOnnxDetector if you intend to use this fp16 model.
        _export_onnx(
            model_path,
            output_path=output_dir / "best.onnx",
            half=True,
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
        )

    elif backend == "all":
        # Export both variants for all backends.
        _export_onnx(
            model_path,
            output_path=output_dir / "best.onnx",
            half=False,
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
        )
        _export_onnx(
            model_path,
            output_path=output_dir / "best_fp16.onnx",
            half=True,
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
        )

    else:
        print(f"ERROR: unknown backend '{backend}'", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLOv8/YOLO11 .pt model to ONNX for ORT backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="assets/v8s.pt",
        help="Path to the PyTorch .pt source model (default: assets/v8s.pt)",
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "cuda", "directml", "all"],
        default="cpu",
        help=(
            "Target inference backend. "
            "'cpu' and 'directml' produce fp32 ONNX; "
            "'cuda' produces fp16 ONNX; "
            "'all' produces both. (default: cpu)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="assets",
        help="Directory to write exported ONNX file(s) (default: assets)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size used during export (default: 640)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable onnxsim graph simplification",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    export_for_backend(
        model_path=model_path,
        backend=args.backend,
        output_dir=Path(args.output_dir),
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
    )


if __name__ == "__main__":
    main()
