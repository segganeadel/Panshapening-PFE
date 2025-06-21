#!/usr/bin/env python3
import argparse
import sys
import torch

# Try to import measure_flops from Lightning; fallback if necessary
def import_measure_flops():
    try:
        # Lightning 2.x
        from lightning.fabric.utilities.throughput import measure_flops
        return measure_flops
    except ImportError:
        try:
            # Some versions might expose via pytorch_lightning
            from pytorch_lightning.utilities.flops import get_model_complexity_info
            # Wrap into a measure_flops-like signature? But get_model_complexity_info expects module and input size.
            # We'll not fallback to this automatically; instead warn user.
            return None
        except ImportError:
            return None

measure_flops = import_measure_flops()

# Import your model class. Assumes models/ is a (namespace) package or has __init__.py.
try:
    from models.mambfuse import MambFuse
except ImportError as e:
    print("Failed to import MambFuse from models.mambfuse. "
          "Make sure this script is run from project root and that models/ is importable.", file=sys.stderr)
    raise

def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable

def estimate_flops_forward(model: torch.nn.Module, dummy_input: dict):
    if measure_flops is None:
        print("measure_flops utility not found in this Lightning installation; cannot estimate FLOPs.", file=sys.stderr)
        return None
    # Create a meta-device copy of the model
    try:
        with torch.device("meta"):
            # Re-instantiate model on meta device
            # WARNING: __init__ of MambFuse may try to use real device; if so, this might error.
            meta_model = MambFuse(spectral_num=args.spectral_num,
                                  channel=getattr(args, "channel", 32),
                                  satellite=args.satellite,
                                  mtf_kernel_size=args.mtf_kernel_size,
                                  ratio=args.ratio)
            # Put parameters to meta
            meta_model.to(device="meta")
            # Prepare dummy tensors on meta device
            # forward expects input dict with 'lms' and 'pan'
            # lms: [batch, spectral_num, H, W]; pan: [batch, 1, H, W]
            bs = 1
            spectral = args.spectral_num
            H, W = args.height, args.width
            dummy_lms = torch.randn(bs, spectral, H, W, device="meta")
            dummy_pan = torch.randn(bs, 1, H, W, device="meta")
            dummy = {'lms': dummy_lms, 'pan': dummy_pan}

        # Define forward function
        def forward_fn():
            return meta_model(dummy)
        # Attempt FLOPs estimation
        flops = measure_flops(meta_model, forward_fn)
        return flops
    except Exception as e:
        print(f"Warning: FLOPs estimation failed: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count parameters and estimate FLOPs of MambFuse model")
    parser.add_argument("--spectral-num", type=int, required=True,
                        help="Number of spectral bands (spectral_num) for MambFuse")
    parser.add_argument("--height", type=int, default=64,
                        help="Height of dummy input for FLOPs estimation (default: 64)")
    parser.add_argument("--width", type=int, default=64,
                        help="Width of dummy input for FLOPs estimation (default: 64)")
    parser.add_argument("--satellite", type=str, default="qb",
                        help="Satellite name argument passed to MambFuse (default: 'qb')")
    parser.add_argument("--mtf-kernel-size", type=int, default=41,
                        help="mtf_kernel_size passed to MambFuse (default: 41)")
    parser.add_argument("--ratio", type=int, default=4,
                        help="ratio passed to MambFuse (default: 4)")
    # channel is internal; default matches model code
    parser.add_argument("--channel", type=int, default=32,
                        help="Internal channel size passed to MambFuse backbone (default: 32)")
    args = parser.parse_args()

    # Instantiate model on real device (CPU)
    model = MambFuse(spectral_num=args.spectral_num,
                     channel=args.channel,
                     satellite=args.satellite,
                     mtf_kernel_size=args.mtf_kernel_size,
                     ratio=args.ratio)
    model.eval()

    # Count parameters
    total, trainable, non_trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {non_trainable:,}")

    # Estimate FLOPs
    flops = estimate_flops_forward(model, None)
    if flops is not None:
        print(f"Estimated FLOPs per forward pass (batch=1): {flops:,}")
    else:
        print("FLOPs estimation was not available or failed.")
