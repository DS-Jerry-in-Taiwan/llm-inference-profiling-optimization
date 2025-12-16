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
git clone <your-repo-url>
cd llm-inference-optimization

# 2. (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run baseline experiment
python src/baseline.py

# 5. Run ONNX optimization
python src/optimize_onnx.py

# 6. Run quantization experiment
python src/optimize_quantization.py

# 7. View results
ls results/charts/
```

---

## Project Structure

- `src/` - All experiment scripts (with docstrings and comments)
- `results/` - JSON metrics and charts
- `models/` - Downloaded and quantized model files
- `docs/` - Technical methodology and deep dives
- `notebooks/` - Jupyter analysis (optional)

---

## Documentation

See [docs/methodology.md](docs/methodology.md) for technical details on KV-Cache, ONNX, and Quantization.

---

## License

MIT License
