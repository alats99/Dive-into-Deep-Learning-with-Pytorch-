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
| Act Fn | Custom (ms) | PyTorch (ms) | Magnitude Diff |
| :--- | :--- | :--- | :--- |
| Sigmoid | 0.136843 | 0.028988 | 4.72x |
| Softmax | 0.070143 | 0.025287 | 2.77x |
