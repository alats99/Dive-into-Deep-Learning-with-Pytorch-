import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
from activation_library import activation_fns

sns.set_style("darkgrid")

class DeepMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, activation_class):
        super().__init__()
        layers = []
        in_features = input_size
        for i in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(activation_class())
            in_features = hidden_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def get_gradients(model, input_data, target):
    model.zero_grad()
    output = model(input_data)
    loss = F.mse_loss(output, target)
    loss.backward()
    
    grads = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            grads.append(param.grad.abs().mean().item())
    return grads

def visualize_gradient_flow(results, num_layers, init_name, save_path="gradient_flow_comparison.png"):
    plt.figure(figsize=(12, 7))
    
    for act_name, grads in results.items():
        plt.plot(range(1, num_layers + 1), grads, marker='o', label=act_name.capitalize())

    plt.yscale('log')
    plt.xlabel('Layer Index (Input to Output)')
    plt.ylabel('Mean Absolute Gradient (Log Scale)')
    plt.title(f'Gradient Flow Comparison ({init_name} init) across {num_layers} Layers')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved to {save_path}")

def update_readme_table(results, init_name, filename="README.md"):
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Skipping table update.")
        return

    # Prepare table header
    title = f"### {init_name.capitalize()} Initialization Results"
    header = "| Activation | Layer 1 (Log10) | Layer 10 (Log10) | Layer 20 (Log10) | Result |\n| :--- | :--- | :--- | :--- | :--- |\n"
    rows = []
    
    for act_name, grads in results.items():
        l1, l10, l20 = grads[0], grads[9], grads[19]
        
        # Calculate log10 safely
        log_l1 = np.log10(l1) if l1 > 0 else -np.inf
        log_l10 = np.log10(l10) if l10 > 0 else -np.inf
        log_l20 = np.log10(l20) if l20 > 0 else -np.inf

        status = "Healthy"
        if l1 < 1e-10: status = "Vanished"
        elif l1 > 1e2: status = "Exploded"
        
        rows.append(f"| {act_name.capitalize()} | {l1:.2e} ({log_l1:.1f}) | {l10:.2e} ({log_l10:.1f}) | {l20:.2e} ({log_l20:.1f}) | {status} |")
    
    table_content = f"\n{title}\n\n{header}" + "\n".join(rows) + "\n"

    with open(filename, "r") as f:
        content = f.read()

    marker = f"<!-- TABLE_{init_name.upper()} -->"
    
    if marker in content:
        parts = content.split(marker)
        if len(parts) >= 3:
            new_content = parts[0] + marker + table_content + marker + parts[2]
        else:
            new_content = content + f"\n{marker}{table_content}{marker}\n"
    else:
        new_content = content.strip() + f"\n\n{marker}{table_content}{marker}\n"

    with open(filename, "w") as f:
        f.write(new_content)
    print(f"README.md updated with {init_name} results.")

def main():
    parser = argparse.ArgumentParser(description="Gradient Flow Investigation")
    parser.add_argument("--init", type=str, default="default", choices=["default", "xavier", "kaiming", "orthogonal"],
                        help="Weight initialization method")
    parser.add_argument("--layers", type=int, default=20, help="Number of layers in MLP")
    args = parser.parse_args()

    input_size = 100
    hidden_size = 256
    num_layers = args.layers
    batch_size = 64
    
    torch.manual_seed(42)
    input_data = torch.randn(batch_size, input_size)
    target = torch.randn(batch_size, hidden_size)
    
    results = {}
    
    print(f"Investigating gradient flow for {num_layers} layers with {args.init} init...")
    
    for name, act_class in activation_fns.items():
        print(f"  Analyzing {name}...")
        model = DeepMLP(input_size, hidden_size, num_layers, act_class)
        
        # Apply initialization
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if args.init == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif args.init == "kaiming":
                    #  Kaiming nonlinearity selection mapping
                    nonlinearity_map = {
                        "sigmoid": "sigmoid",
                        "tanh": "tanh",
                        "relu": "relu",
                        "leaky_relu": "leaky_relu",
                        "elu": "relu",
                        "swish": "relu",
                        "gelu": "relu",
                        "mish": "relu",
                        "selu": "linear",
                        "softplus": "sigmoid",
                        "hardswish": "relu",
                        "sine": "linear" 
                    }
                    gain_key = nonlinearity_map.get(name, "relu")
                    nn.init.kaiming_normal_(m.weight, nonlinearity=gain_key)
                elif args.init == "orthogonal":
                    nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0) # default  bias initialization
        
        grads = get_gradients(model, input_data, target)
        results[name] = grads

    visualize_gradient_flow(results, num_layers, args.init)
    update_readme_table(results, args.init)

if __name__ == "__main__":
    main()
