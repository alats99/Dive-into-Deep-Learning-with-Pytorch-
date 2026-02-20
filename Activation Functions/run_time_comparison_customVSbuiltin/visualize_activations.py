import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import time
import os

sns.set_style("darkgrid")


def visualize(x, custom_output, pytorch_output, title : str = f"ADD NAME"):
    x_np = x.detach().cpu().numpy()
    custom_np = custom_output.detach().cpu().numpy()
    pytorch_np = pytorch_output.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(x_np, custom_np, color="coral", linewidth=2)
    axes[0].set_title(f"Custom {title}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(f"{title}(x)")

    axes[1].plot(x_np, pytorch_np, color="dodgerblue", linewidth=2)
    axes[1].set_title(f"PyTorch {title}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(f"{title}(x)")

    plt.suptitle(f"{title} Activation — Custom vs PyTorch", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{title.lower()}_comparison.png", dpi=150)
    plt.show()
    print(f"Plot saved to {title.lower()}_comparison.png")


def benchmark(fn, x, label, n_runs=1000):
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = fn(x)
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / n_runs) * 1000
    print(f"  {label}: {avg_ms:.4f} ms/run  (total {elapsed:.2f}s over {n_runs} runs)")
    return avg_ms

def export_to_readme(results, filename="README.md"):
    # Export results to README
    name = results[0]['name']
    custom_time = [r['time'] for r in results if r['type'] == 'Custom'][0]
    pytorch_time = [r['time'] for r in results if r['type'] == 'PyTorch'][0]
    magnitude_diff = custom_time / pytorch_time
    
    header = "## Benchmark results:\n| Act Fn | Custom (ms) | PyTorch (ms) | Magnitude Diff |\n| :--- | :--- | :--- | :--- |\n"
    new_row = f"| {name} | {custom_time:.6f} | {pytorch_time:.6f} | {magnitude_diff:.2f}x |\n"
    
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write(header + new_row)
        return

    with open(filename, "r") as f:
        content = f.read()

    if "## Benchmark results:" in content:
        # Check if this function already has a row
        if f"| {name} |" in content:
            # Replace existing row
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if line.startswith(f"| {name} |"):
                    lines[i] = new_row.strip()
                    break
            new_content = "\n".join(lines)
        else:

            new_content = content.strip() + "\n" + new_row
    else:
        new_content = content.strip() + "\n\n" + header + new_row

    with open(filename, "w") as f:
        f.write(new_content)
    
    print(f"Results exported to {filename}")