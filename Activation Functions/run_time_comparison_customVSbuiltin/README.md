# Activation Function Comparison

Manual implementation of activation functions versus PyTorch optimized built-ins.

## Project Structure
- `activation_fn_library.py`: Manual Sigmoid and Softmax classes.
- `main.py`: Dynamic benchmarking and comparison script.
- `dataset.py`: Continuous 1D data generation.
- `visualize_activations.py`: Plotting and benchmarking utilities.
- `set_seed.py`: Seed management for reproducibility.

## Usage
Run the script by specifying the class name from the library:
```bash
python main.py --act_fn Sigmoid
python main.py --act_fn Softmax
```

## Highlights
- Reflection: Uses `getattr` to dynamically load classes and find PyTorch equivalents.
- OOP: Base `Activation` class handles naming and the `forward` pass.
- Verification: Automatic speed benchmarking and accuracy checks.

## Benchmark results:
| Act Fn | Custom (ms) | PyTorch (ms) |
| :--- | :--- | :--- |
| Sigmoid | 0.133350 | 0.047236 |
| Softmax | 0.083846 | 0.028065 |
