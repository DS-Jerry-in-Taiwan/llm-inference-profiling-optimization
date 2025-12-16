"""Environment verification script for LLM Inference Profiling & Optimization.

Checks:
- Python version
- PyTorch installation and version
- CUDA availability
- Transformers installation
- ONNX Runtime installation
"""

import sys
import importlib

def check_python():
    print("Python version:", sys.version.replace('\n', ' '))
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required.")
    else:
        print("✅ Python version OK.")

def check_torch():
    try:
        import torch
        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        print("✅ PyTorch installed.")
    except ImportError:
        print("❌ PyTorch not installed.")

def check_transformers():
    try:
        import transformers
        print("Transformers version:", transformers.__version__)
        print("✅ Transformers installed.")
    except ImportError:
        print("❌ Transformers not installed.")

def check_onnxruntime():
    try:
        import onnxruntime
        print("ONNX Runtime version:", onnxruntime.__version__)
        print("✅ ONNX Runtime installed.")
    except ImportError:
        print("❌ ONNX Runtime not installed.")

def check_accelerate():
    try:
        import accelerate
        print("Accelerate version:", accelerate.__version__)
        print("✅ Accelerate installed.")
    except ImportError:
        print("❌ Accelerate not installed.")

if __name__ == "__main__":
    print("=== Environment Check: LLM Inference Profiling & Optimization ===")
    check_python()
    print()
    check_torch()
    print()
    check_transformers()
    print()
    check_onnxruntime()
    print()
    check_accelerate()
