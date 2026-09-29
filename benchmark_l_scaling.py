#!/usr/bin/env python3
"""Measure FP32 single-forward peak memory across residue lengths."""

# a4-l-scaling:start
import argparse
import gc
import json
from pathlib import Path

import torch

from mdgen.rigid_utils import Rigid
from mdgen.wrapper import NewMDGenWrapper


DEFAULT_LENGTHS = (256, 1000, 2500, 5000, 7500)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark one full MDGen model forward at each residue length."
    )
    parser.add_argument("--sim_ckpt", required=True, help="Path to atlas.ckpt.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path.")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=DEFAULT_LENGTHS,
        help="Residue lengths to measure.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=250,
        help="Trajectory frames T; Atlas inference uses 250.",
    )
    parser.add_argument("--use_sdpa", action="store_true")
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument(
        "--continue_after_oom",
        action="store_true",
        help="Continue to larger lengths after an OOM (normally unnecessary).",
    )
    return parser.parse_args()


def write_report(path, report):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


def attention_paths(model):
    paths = {}
    for name, module in model.named_modules():
        if hasattr(module, "last_attention_backend"):
            paths[name] = {
                "path": module.last_attention_backend,
                "fallback_reason": module.last_sdpa_fallback_reason,
            }
    return paths


def make_forward_inputs(model, length, num_frames, device):
    batch_size = 1
    latent_dim = model.latent_dim
    x = torch.zeros(
        batch_size,
        num_frames,
        length,
        latent_dim,
        dtype=torch.float32,
        device=device,
    )
    mask = torch.ones(
        batch_size, num_frames, length, dtype=torch.float32, device=device
    )
    x_cond = torch.zeros_like(x)
    x_cond_mask = torch.zeros(
        batch_size, num_frames, length, dtype=torch.long, device=device
    )
    if model.args.sim_condition:
        x_cond_mask[:, 0] = 1

    return {
        "x": x,
        "t": torch.full((batch_size,), 0.5, dtype=torch.float32, device=device),
        "mask": mask,
        "start_frames": Rigid.identity(
            (batch_size, length),
            dtype=torch.float32,
            device=device,
            requires_grad=False,
            fmt="rot_mat",
        ),
        "end_frames": None,
        "x_cond": x_cond,
        "x_cond_mask": x_cond_mask,
        "aatype": torch.zeros(
            batch_size, length, dtype=torch.long, device=device
        ),
        "position_ids": torch.arange(
            length, dtype=torch.long, device=device
        ).unsqueeze(0),
    }


def memory_stats():
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    return {
        "peak_memory_allocated_bytes": allocated,
        "peak_memory_allocated_gib": allocated / 1024**3,
        "peak_memory_reserved_bytes": reserved,
        "peak_memory_reserved_gib": reserved / 1024**3,
    }


def run_case(model, length, num_frames, device):
    inputs = None
    output = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()

    try:
        inputs = make_forward_inputs(model, length, num_frames, device)
        with torch.inference_mode():
            output = model.model(**inputs)
        torch.cuda.synchronize()
        return {
            "length": length,
            "oom": False,
            "baseline_memory_allocated_bytes": baseline_allocated,
            "baseline_memory_allocated_gib": baseline_allocated / 1024**3,
            **memory_stats(),
            "output_shape": list(output.shape),
            "attention_modules": attention_paths(model.model),
        }
    except RuntimeError as error:
        if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in str(error).lower():
            raise
        return {
            "length": length,
            "oom": True,
            "peak_is_pre_oom_lower_bound": True,
            "baseline_memory_allocated_bytes": baseline_allocated,
            "baseline_memory_allocated_gib": baseline_allocated / 1024**3,
            **memory_stats(),
            "error": str(error),
        }
    finally:
        del output
        del inputs
        gc.collect()
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A4 L-scaling requires a CUDA GPU")
    if args.num_frames <= 0 or any(length <= 0 for length in args.lengths):
        raise ValueError("--num_frames and every --lengths value must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda", torch.cuda.current_device())
    model = NewMDGenWrapper.load_from_checkpoint(
        args.sim_ckpt,
        use_sdpa=args.use_sdpa,
        weights_only=False,
    )
    model.eval().float().to(device)
    model.args.grad_checkpointing = False

    report = {
        "benchmark": "A4 L-scaling single model forward",
        "checkpoint": str(Path(args.sim_ckpt).resolve()),
        "measurement_scope": (
            "Full LatentMDGenModel forward in torch.inference_mode; peak includes "
            "model parameters, synthetic inputs, and forward temporaries, but excludes "
            "checkpoint-loading peak."
        ),
        "attention_path": "sdpa" if args.use_sdpa else "manual",
        "dtype": "float32",
        "batch_size": 1,
        "num_frames": args.num_frames,
        "grad_checkpointing": False,
        "deterministic": True,
        "cudnn_benchmark": False,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "cuda_wheel": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "lengths_requested": args.lengths,
        "results": [],
    }
    write_report(args.output, report)

    for length in args.lengths:
        print(f"A4: attention={report['attention_path']} B=1 T={args.num_frames} L={length}")
        result = run_case(model, length, args.num_frames, device)
        report["results"].append(result)
        write_report(args.output, report)
        status = "OOM" if result["oom"] else "success"
        print(
            f"A4: L={length} {status}, "
            f"peak_allocated={result['peak_memory_allocated_gib']:.3f} GiB"
        )
        if result["oom"] and not args.continue_after_oom:
            break

    print(f"A4 report saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
# a4-l-scaling:end
