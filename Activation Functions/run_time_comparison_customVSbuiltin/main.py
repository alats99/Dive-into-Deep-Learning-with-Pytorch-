# Comparison file
# We import all the manually created activation functions and compare their running speed
# with the pytorch implementations
# Experiments are implemented on M4 mps gpu.

import torch
import torch.nn as nn
import torch.utils.data as data
import argparse
import sys

# Import everything from the library
import activation_fn_library as lib
from dataset import ContinuousData
from set_seed import set_seed
from visualize_activations import visualize, benchmark, export_to_readme


def main(custom_fn):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    set_seed(42, device=str(device))
    print(f"Using {device} device")
    print(f"Running comparison for: {custom_fn.name}")

    # Continuous data generations
    dataset = ContinuousData(size=1000, low=-10.0, high=10.0)
    loader = data.DataLoader(dataset, batch_size=len(dataset))
    x = next(iter(loader)).squeeze(1).to(device)  # shape: (1000,)

    # Custom implementation
    custom_fn = custom_fn.to(device)
    custom_output = custom_fn(x)

    # find a matching class
    try:
        pytorch_cls = getattr(nn, custom_fn.name)

        if custom_fn.name == "Softmax":
            pytorch_fn = pytorch_cls(dim=-1)         # Softmax needs a dim argument
        else:
            pytorch_fn = pytorch_cls()
    except AttributeError:
        print(f"Error: Could not find a PyTorch built-in named 'nn.{custom_fn.name}'")
        return

    pytorch_fn = pytorch_fn.to(device)
    pytorch_output = pytorch_fn(x)

    # Max difference
    max_diff = (custom_output - pytorch_output).abs().max().item()
    print(f"Max difference (custom vs PyTorch): {max_diff:.2e}")
    
    if max_diff < 1e-6:
        print(f"Custom {custom_fn.name} matches PyTorch implementation")
    else:
        print(f"Warning: Results differ! Max diff: {max_diff}")

    # Speed comparison
    print(f"\nBenchmarking {custom_fn.name} (1000 runs):")
    custom_time = benchmark(custom_fn, x, f"Custom {custom_fn.name}")
    pytorch_time = benchmark(pytorch_fn, x, f"PyTorch {custom_fn.name}")

    # Export to README
    results = [
        {"name": custom_fn.name, "type": "Custom", "time": custom_time},
        {"name": custom_fn.name, "type": "PyTorch", "time": pytorch_time}
    ]
    export_to_readme(results)

    # Visualize 
    visualize(x.cpu(), custom_output.cpu(), pytorch_output.cpu(), title=custom_fn.name)
    print("-" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare custom activation functions with PyTorch built-ins.")
    parser.add_argument(
        "--act_fn",
        default = "Sigmoid",
        type=str, 
        required=False,
        help="The name of the class in activation_fn_library (Case Sensitive, e.g., Sigmoid)"
    )
    
    args = parser.parse_args()
    
    try:
        cls = getattr(lib, args.act_fn)
        selected_fn = cls()
        main(selected_fn)
    except AttributeError:
        print(f"Error: Class '{args.act_fn}' not found in activation_fn_library.py")
        print("Available classes are:", [name for name in dir(lib) if not name.startswith("__")]) # reject the dunder names,
