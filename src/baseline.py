"""
BaselineInference: Minimal LLM inference baseline with/without KV-cache.

Implements:
- BaselineInference class (load_model, generate, measure_latency)
- Experiment: GPT2, prompt, latency measurement, results
"""

import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import os

class BaselineInference:
    """
    BaselineInference provides minimal LLM inference with and without KV-cache.

    Methods:
        load_model(): Loads GPT2 model and tokenizer to device.
        generate(prompt, max_new_tokens, use_cache): Generates text with/without KV-cache.
        measure_latency(prompt, max_new_tokens, use_cache, num_runs): Measures prefill, decode, and total latency.
    """
    def __init__(self, model_name="gpt2"):
        """
        Initialize BaselineInference.

        Args:
            model_name (str): Model name to load from HuggingFace.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Loads the model and tokenizer to the selected device.
        """
        print(f"Loading model: {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def generate(self, prompt, max_new_tokens=50, use_cache=True):
        """
        Generate text from a prompt using the model.

        Args:
            prompt (str): Input prompt.
            max_new_tokens (int): Number of tokens to generate.
            use_cache (bool): Whether to use KV-cache.

        Returns:
            str: Generated text.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                do_sample=False,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def measure_latency(self, prompt, max_new_tokens=50, use_cache=True, num_runs=3):
        """
        Measure prefill, decode, and total latency for LLM inference.

        Args:
            prompt (str): Input prompt.
            max_new_tokens (int): Number of tokens to generate.
            use_cache (bool): Whether to use KV-cache.
            num_runs (int): Number of runs to average.

        Returns:
            dict: Latency statistics.
        """
        prefill_latencies = []
        decode_latencies = []
        total_latencies = []

        for _ in range(num_runs):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Prefill (first token)
            start = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)
            prefill_time = time.perf_counter() - start

            # Decode (remaining tokens)
            generated = input_ids
            past_key_values = outputs.past_key_values if use_cache else None
            decode_times = []
            for _ in range(max_new_tokens):
                start = time.perf_counter()
                with torch.no_grad():
                    if use_cache and past_key_values is not None:
                        outputs = self.model(input_ids=generated[:, -1:], past_key_values=past_key_values, use_cache=use_cache)
                        past_key_values = outputs.past_key_values
                    else:
                        outputs = self.model(input_ids=generated, use_cache=use_cache)
                decode_time = time.perf_counter() - start
                decode_times.append(decode_time)
                # Append next token
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
            total_time = prefill_time + sum(decode_times)
            prefill_latencies.append(prefill_time)
            decode_latencies.append(sum(decode_times) / max_new_tokens)
            total_latencies.append(total_time)

        return {
            "prefill_latency_avg": sum(prefill_latencies) / num_runs,
            "decode_latency_per_token_avg": sum(decode_latencies) / num_runs,
            "total_latency_avg": sum(total_latencies) / num_runs,
            "prefill_latencies": prefill_latencies,
            "decode_latencies": decode_latencies,
            "total_latencies": total_latencies,
        }

def main():
    prompt = "The future of artificial intelligence is"
    max_new_tokens = 50
    num_runs = 3
    model_name = "gpt2"

    os.makedirs("results/charts", exist_ok=True)

    baseline = BaselineInference(model_name)
    baseline.load_model()

    print("Running without KV-cache...")
    no_cache = baseline.measure_latency(prompt, max_new_tokens, use_cache=False, num_runs=num_runs)
    print("Running with KV-cache...")
    with_cache = baseline.measure_latency(prompt, max_new_tokens, use_cache=True, num_runs=num_runs)

    results = {
        "no_cache": no_cache,
        "with_cache": with_cache,
        "config": {
            "model": model_name,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "num_runs": num_runs,
            "device": baseline.device,
        }
    }

    # Save results
    with open("results/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot comparison
    labels = ["No KV-Cache", "With KV-Cache"]
    total_times = [no_cache["total_latency_avg"], with_cache["total_latency_avg"]]
    per_token = [no_cache["decode_latency_per_token_avg"], with_cache["decode_latency_per_token_avg"]]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(labels, total_times, color=["red", "green"])
    ax[0].set_title("Total Latency (s)")
    ax[0].set_ylabel("Seconds")

    ax[1].bar(labels, per_token, color=["red", "green"])
    ax[1].set_title("Per Token Decode Latency (s)")
    ax[1].set_ylabel("Seconds")

    plt.tight_layout()
    plt.savefig("results/charts/baseline_comparison.png")
    print("Baseline experiment completed successfully.")

if __name__ == "__main__":
    main()
