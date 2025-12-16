"""
ONNX Optimization Engineer: Export GPT2 to ONNX and benchmark with ONNX Runtime.

- Tries optimum.onnxruntime for export (KV-cache auto-handling).
- Falls back to torch.onnx.export if optimum not available.
- Runs ONNX inference and compares latency to PyTorch baseline.
- Generates onnx_comparison.png and prints speedup.
"""

import os
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def export_onnx_optimum(model_name, onnx_path):
    try:
        from optimum.exporters.onnx import main_export
        from argparse import Namespace
        # Use optimum.exporters.onnx.main_export for CLI-like export
        args = Namespace(
            model=model_name,
            output=os.path.dirname(onnx_path),
            task="causal-lm",
            device="cpu",
            fp16=False,
            optimize=None,
            batch_size=1,
            sequence_length=8,
            no_post_process=False,
            trust_remote_code=False,
            pad_token_id=None,
            for_ort=False,
            use_subprocess=False,
            cache_dir=None,
            revision=None,
            framework="pt",
            do_validation=False,
        )
        main_export(args)
        print("Exported ONNX using optimum.exporters.onnx.")
        return True
    except ImportError:
        print("optimum not installed, falling back to torch.onnx.export.")
        return False

def export_onnx_torch(model_name, onnx_path):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    dummy_input = tokenizer("The future of artificial intelligence is", return_tensors="pt")["input_ids"]

    # Define a wrapper to only output logits (no cache)
    class LogitsOnly(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_ids):
            return self.model(input_ids=input_ids, use_cache=False).logits

    wrapper = LogitsOnly(model)
    torch.onnx.export(
        wrapper,
        (dummy_input,),
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}},
        opset_version=13,
    )
    print("Exported ONNX using torch.onnx.export (logits only, no cache).")

class OnnxInference:
    """
    ONNX Runtime inference wrapper for GPT2.

    Methods:
        generate(input_ids, max_new_tokens): Generate tokens using ONNX model.
        measure_latency(input_ids, max_new_tokens, num_runs): Benchmark ONNX inference latency.
    """
    def __init__(self, onnx_path):
        """
        Initialize ONNX Runtime session.

        Args:
            onnx_path (str): Path to ONNX model file.
        """
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def generate(self, input_ids, max_new_tokens=50):
        """
        Generate tokens using ONNX Runtime.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            max_new_tokens (int): Number of tokens to generate.

        Returns:
            torch.Tensor: Generated token IDs.
        """
        generated = input_ids
        for _ in range(max_new_tokens):
            ort_inputs = {self.input_name: generated.cpu().numpy()}
            logits = self.session.run([self.output_name], ort_inputs)[0]
            next_token = np.argmax(logits[:, -1, :], axis=-1)
            next_token = np.array(next_token).reshape(-1, 1)
            generated = np.concatenate([generated.cpu().numpy(), next_token], axis=1)
            generated = torch.tensor(generated, dtype=torch.long)
        return generated

    def measure_latency(self, input_ids, max_new_tokens=50, num_runs=3):
        """
        Measure total latency for ONNX inference.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            max_new_tokens (int): Number of tokens to generate.
            num_runs (int): Number of runs to average.

        Returns:
            dict: Latency statistics.
        """
        total_latencies = []
        for _ in range(num_runs):
            gen_ids = input_ids.clone()
            start = time.perf_counter()
            self.generate(gen_ids, max_new_tokens)
            total_latencies.append(time.perf_counter() - start)
        return {
            "total_latency_avg": sum(total_latencies) / num_runs,
            "total_latencies": total_latencies,
        }

def main():
    model_name = "gpt2"
    onnx_path = "models/gpt2.onnx"
    os.makedirs("models", exist_ok=True)

    # Download ONNX model from HuggingFace Hub if export fails
    try:
        exported = export_onnx_optimum(model_name, onnx_path)
        if not exported:
            export_onnx_torch(model_name, onnx_path)
    except Exception:
        print("ONNX export failed, downloading pre-converted model from HuggingFace Hub...")
        from huggingface_hub import hf_hub_download
        onnx_path = hf_hub_download(
            repo_id="onnx/models",
            subfolder="gpt2-lm-head",
            filename="model.onnx",
            cache_dir="models"
        )
        print(f"Downloaded ONNX model to {onnx_path}")

    # Prepare input
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # ONNX inference
    onnx_inf = OnnxInference(onnx_path)
    onnx_result = onnx_inf.measure_latency(input_ids, max_new_tokens=50, num_runs=3)

    # Load PyTorch baseline
    import json
    with open("results/baseline_results.json") as f:
        baseline = json.load(f)
    pt_latency = baseline["no_cache"]["total_latency_avg"]
    onnx_latency = onnx_result["total_latency_avg"]
    speedup = pt_latency / onnx_latency if onnx_latency > 0 else 0

    # Save ONNX results
    with open("results/onnx_results.json", "w") as f:
        json.dump(onnx_result, f, indent=2)

    # Plot comparison
    import matplotlib.pyplot as plt
    labels = ["PyTorch (No-Cache)", "ONNX (No-Cache)"]
    times = [pt_latency, onnx_latency]
    fig, ax = plt.subplots()
    ax.bar(labels, times, color=["blue", "orange"])
    ax.set_ylabel("Total Latency (s)")
    ax.set_title("PyTorch vs. ONNX Inference Latency")
    for i, v in enumerate(times):
        ax.text(i, v, f"{v:.2f}s", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig("results/charts/onnx_comparison.png")
    print(f"ONNX Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()
