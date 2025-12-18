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

- `src/` - All experiment scripts (with docstrings and comments)
- `results/` - JSON metrics and charts
- `models/` - Downloaded and quantized model files
- `docs/` - Technical methodology and deep dives
- `notebooks/` - Jupyter analysis (optional)

---

## Documentation

See [docs/methodology.md](docs/methodology.md) for technical details on KV-Cache, ONNX, and Quantization.

---

## Core Performance Engineering Workflow

本專案的效能優化與分析流程如下：

1. **定義使用場景與需求**  
   明確你的應用目標與效能需求。

2. **歸類到四大核心類別**  
   - 使用者體驗 (UX)
   - 系統容量 (Capacity)
   - 資源效率 (Efficiency)
   - 穩定性 (Reliability)

3. **針對歸類挑選指標**  
   例如：UX 關注延遲，Efficiency 關注資源用量等。

4. **把指標寫到 baseline 做資料收集**  
   以 `baseline.py` 收集各種優化前後的指標數據。

5. **把指標定義到 profiler 並分析推論模組效能**  
   以 profiler 針對細部 operator 做分析，找出瓶頸。

6. **針對分析找出瓶頸**  
   這是效能優化的核心。

7. **優化瓶頸**  
   例如用 KV-cache、ONNX、量化等技術。

8. **透過 baseline 驗證優化結果**  
   再次量測指標，確認優化是否有效。

---

## License

MIT License
