# LLM Inference Profiling & Optimization

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![ONNX](https://img.shields.io/badge/ONNX-Supported-green)

## Executive Summary

Achieved up to **2.6x speedup** on GPT-2 inference using Quantization and ONNX optimization. This project demonstrates systematic performance engineering for LLMs, including baseline, KV-Cache, ONNX, and INT8 quantization.

---

## Key Results

| Method         | Latency (50 tokens) | Speedup | Model Size |
|----------------|---------------------|---------|------------|
| Baseline       | 1.65s               | 1.0x    | 522MB      |
| KV-Cache       | 0.71s               | 2.3x    | -          |
| ONNX           | 0.75s               | 2.2x    | -          |
| INT8 Quant     | 0.64s               | 2.6x    | 511MB      |

---

## Final Comparison

![Final Comparison](results/charts/final_comparison.png)

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/DS-Jerry-in-Taiwan/llm-inference-profiling-optimization.git
cd llm-inference-profiling-optimization

# 2. (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run environment check (optional, recommended)
python setup_check.py

# 5. Run baseline experiment
python src/baseline.py

# 6. Run ONNX optimization
python src/optimize_onnx.py

# 7. Run quantization experiment
python src/optimize_quantization.py

# 8. View results (charts and JSON)
ls results/charts/
ls results/*.json

# 9. Open charts (example)
# On Linux/macOS:
xdg-open results/charts/final_comparison.png
# On Windows:
start results/charts/final_comparison.png
```

---

## Project Structure

Project directory structure and the purpose of each path/file:

```
llm-inference-optimization/
├── README.md                # Project overview, quick start, workflow, and results
├── requirements.txt         # Python dependencies
├── setup_check.py           # Environment check script
├── docs/
│   └── methodology.md       # Technical methodology and details
├── models/
│   ├── gpt2_int8.pt         # Quantized GPT2 PyTorch weights
│   ├── gpt2.onnx            # Converted ONNX model
│   └── ...                  # Other model files and configs
├── notebooks/
│   └── analysis.ipynb       # Jupyter Notebook for interactive analysis (optional)
├── results/
│   ├── baseline_results.json        # Baseline experiment results
│   ├── onnx_results.json            # ONNX experiment results
│   ├── quantization_results.json    # Quantization experiment results
│   ├── profiling_report.md          # Profiler analysis report
│   ├── charts/
│   │   └── ...                      # Performance comparison charts
│   └── traces/
│       ├── trace_no_cache.json      # Profiler trace (no KV-cache)
│       └── trace_with_cache.json    # Profiler trace (with KV-cache)
├── src/
│   ├── __init__.py                  # Python module initialization
│   ├── baseline.py                  # Baseline inference and latency measurement
│   ├── optimize_onnx.py             # ONNX export and performance testing
│   ├── optimize_quantization.py     # Quantization and accuracy/performance testing
│   ├── profiling.py                 # Profiler deep analysis and bottleneck report
│   ├── modeling.py                  # (Extensible) model-related utilities
│   ├── optimization.py              # (Extensible) optimization utilities
│   └── utils.py                     # (Extensible) general helper functions
```

- **README.md**: Project overview, usage instructions, workflow, and results.
- **requirements.txt**: All required Python packages.
- **setup_check.py**: Checks if your environment meets project requirements.
- **docs/methodology.md**: Technical details, methodology, and design trade-offs.
- **models/**: Stores all model files (original, ONNX, quantized, etc.).
- **notebooks/**: Jupyter Notebooks for interactive analysis and visualization.
- **results/**: All experiment results (JSON, charts, traces, reports).
  - **charts/**: Performance comparison charts (latency, throughput, etc.).
  - **traces/**: PyTorch Profiler traces for Chrome Trace Viewer.
- **src/**: All core scripts and code.
  - **baseline.py**: Baseline inference and latency measurement.
  - **optimize_onnx.py**: ONNX export and performance testing.
  - **optimize_quantization.py**: Quantization and accuracy/performance testing.
  - **profiling.py**: PyTorch Profiler deep analysis and bottleneck reporting.
  - Other files: Extensible for custom tools or helper functions.

---

## Documentation

See [docs/methodology.md](docs/methodology.md) for technical details on KV-Cache, ONNX, and Quantization.

---

## Core Performance Engineering Workflow

This project’s performance optimization and analysis workflow:

1. **Define use cases and requirements**  
   Clearly specify your application goals and performance needs.

2. **Classify into four core categories**  
   - User Experience (UX)
   - System Capacity
   - Resource Efficiency
   - Reliability

3. **Select metrics based on classification**  
   E.g., UX focuses on latency, Efficiency on resource usage, etc.

4. **Implement metrics in the baseline for data collection**  
   Use `baseline.py` to collect metrics before and after optimization.

5. **Define metrics in the profiler and analyze inference performance**  
   Use the profiler to analyze operator-level details and identify bottlenecks.

6. **Identify bottlenecks based on analysis**  
   This is the core of performance optimization.

7. **Optimize bottlenecks**  
   E.g., using KV-cache, ONNX, quantization, etc.

8. **Validate optimization results with the baseline**  
   Re-measure metrics to confirm the effectiveness of optimizations.

---

## License

MIT License
