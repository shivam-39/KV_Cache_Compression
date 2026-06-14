"""
AQuA-CoT evaluation for the TrueCompression method (GearLlamaForCausalLMNew / GearMistralForCausalLMNew).

NOTE: TrueCompression supports Llama and Mistral-architecture models.
      Pass a local or HF Llama or Mistral checkpoint with --model.
      The script automatically detects the model architecture.

This script is based on evaluation_aqua_cot.py and evaluation_gsm8k_true_compression.py:
  - Uses GearLlamaForCausalLMNew for Llama models or GearMistralForCausalLMNew for Mistral models
  - Builds compress_config as a plain dict (no copy_for_all_attention / calculate_compress_ratio_list)
  - Adds --stream, --buffer_len, --compress_mode args for the new cache path
  - Auto-selects device: cuda → mps → cpu
  - Auto-detects model architecture (Llama or Mistral) and uses appropriate model class
  - Writes output JSON to evaluation_aqua_cot_true_compression.json to avoid clobbering Simulated results
"""

import argparse
import json
import os
import logging
import sys
import torch
import numpy as np
import datasets
import accelerate
import transformers
from tqdm.auto import tqdm
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, cast
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from GEARLM import GearLlamaForCausalLMNew
from GEARLM import GearMistralForCausalLMNew
from transformers import AutoTokenizer

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    logging.warning("matplotlib not available; rank distribution plot will be skipped.")

def preprocess_aqua_dataset(dataset):
    def add_text_options(example):
        example["text_options"] = "\n".join(example["options"])
        return example
    processed_dataset = dataset.map(add_text_options)
    return processed_dataset

def prepare_aqua_cot_prompts_and_targets(batch: dict, prompt_prefix:str=""):
    questions = batch["question"]
    options = batch["text_options"]
    prompts = [
        prompt_prefix + "\nQuestion: " + question + "\nOption:\n" + option + "\nLet's think step by step\n" \
            for question,option in zip(questions,options)
    ]
    targets = batch["correct"]
    return prompts, targets

def extract_aqua_answer(generation):
    if "answer is" in generation:
        generation = generation.split("answer is")[1]
    generation = generation.split("\nQuestion:")[0].strip()
    return generation

def evaluate_aqua_answer_cot(generation, target, prompt):
    target_option = target + ")"
    target_answer = prompt.split("\nQuestion:")[-1].split("\nOption:")[1].split("\nLet's think step by step\n")[0].split(target+")")[-1].split("\n")[0]
    pred = extract_aqua_answer(generation)
    if target_answer in pred: 
        is_pred_true = (
            (target_answer in pred) and not all([option+")" in pred for option in ["A","B","C","D"]])
        )
        return pred, target_answer, is_pred_true
    else:
        is_pred_true = (
            (target_option in pred) and not all([option+")" in pred for option in ["A","B","C","D"]])
        )
        return pred, target_option, is_pred_true

@dataclass(frozen=True)
class EvaluationSample:
    question: str
    generation: str
    target: str
    pred: str
    label: str
    is_pred_true: bool

@dataclass(frozen=True)
class EvaluationMetrics(DataClassJsonMixin):
    accuracy: float

@dataclass(frozen=True)
class EvaluationResults(DataClassJsonMixin):
    samples: list
    metrics: EvaluationMetrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AQuA-CoT – TrueCompression")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Model name or path.")
    parser.add_argument("--prompt_file", type=str, default="lib_prompt/aqua/cot_prompt_8shots.txt", help="Prompt file.")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--example_subset", type=str, default=None, help="Dataset slice, e.g. '0:10' or None for full test set.")
    parser.add_argument("--max_length", type=int, default=None, help="")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="")
    parser.add_argument("--model_max_length", type=int, default=4096, help="")
    parser.add_argument("--do_sample", action="store_true", default=False, help="")
    parser.add_argument("--temperature", type=float, default=0.8, help="")
    parser.add_argument("--top_k", type=int, default=50, help="")
    parser.add_argument("--top_p", type=float, default=0.95, help="")
    parser.add_argument("--dataset_split", type=str, default="test", help="")
    parser.add_argument("--root_output_dir", type=str, default="outputs", help="Root output dir")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--debug", action="store_true", default=False, help="")
    # TrueCompression-specific args
    parser.add_argument("--compress_method", type=str, default="None", help="'None' to disable compression, or e.g. 'gear', 'uniform', 'outlier'.")
    parser.add_argument("--compress_mode", type=str, default="gear", help="Compress mode passed to CompressedUnion (gear / uniform / outlier / …).")
    parser.add_argument("--quantize_bit", type=int, default=8, help="")
    parser.add_argument("--rank", type=float, default=0.0, help="")
    parser.add_argument("--loop", type=int, default=0, help="")
    parser.add_argument("--left", type=float, default=0.0, help="")
    parser.add_argument("--buffer_len", type=int, default=20, help="Tail buffer length before compress() flushes (for new method).")
    parser.add_argument("--stream", action="store_true", default=False, help="Use StreamCompressedCache instead of CompressedCache.")
    parser.add_argument("--streaming_gap", type=int, default=1, help="Recompress interval for streaming cache.")
    parser.add_argument("--sink_tokens", type=int, default=4, help="Number of initial prefill tokens always kept in full precision.")
    parser.add_argument("--recency_tokens", type=int, default=64, help="Size of the rolling recency FP window.")
    args = parser.parse_args()

    # Output paths
    root_output_dir = Path(args.root_output_dir)
    output_dir = f"cot_aqua_cot_true_compression"
    if args.example_subset is not None:
        output_dir += f"_subset-{args.example_subset}"
    output_dir = root_output_dir / f"{args.model.split('/')[-1]}" / output_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    results_file = output_dir / f"evaluation_aqua_cot_true_compression.json"

    # Dataset
    split = args.dataset_split if args.example_subset is None else f"{args.dataset_split}[{args.example_subset}]"
    dataset = load_dataset("aqua_rat", split=split)
    eval_dataset = preprocess_aqua_dataset(dataset)
    dataloader = torch.utils.data.DataLoader(
        cast(torch.utils.data.Dataset, eval_dataset),
        batch_size=args.batch_size,
    )

    # Logging
    tb_writter = SummaryWriter(log_dir=str(output_dir.resolve()))
    logging.basicConfig(
        filename=os.path.join(output_dir.resolve(), "log.txt"),
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # Device selection: cuda > mps > cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Build compress_config dict (for new method)
    if args.compress_method == "None":
        compress_config = None
    else:
        compress_config = {
            "compress_mode": args.compress_mode,
            "quantize_bit": args.quantize_bit,
            "rank": args.rank,
            "loop": args.loop,
            "left": args.left,
            "buffer_len": args.buffer_len,
            "stream": args.stream,
            "streaming_gap": args.streaming_gap,
            "sink_tokens": args.sink_tokens,
            "recency_tokens": args.recency_tokens,
        }
    logging.info(f"compress_config: {compress_config}")

    # Import rank tracking functions from TrueCompressFunction
    try:
        from GEARLM.TrueCompression.models.rank_tracker import (
            clear_rank_distribution,
            get_rank_distribution,
            save_rank_distribution_to_csv,
        )
        # Clear any previous rank distribution data
        clear_rank_distribution()
        logging.info("Initialized rank distribution tracking")
    except Exception as exc:
        logging.warning(
            "Could not initialize rank distribution tracking: %s",
            exc,
        )
        get_rank_distribution = None
        save_rank_distribution_to_csv = None

    # Model + tokenizer
    if device.type == "cuda":
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "cache_dir": "../cache",
        }
    else:
        model_kwargs = {
            "torch_dtype": torch.float16,
            "cache_dir": "../cache",
        }
    if args.hf_token is not None:
        model_kwargs["token"] = args.hf_token

    config = transformers.AutoConfig.from_pretrained(
        args.model,
        use_flash_attn=False,
        trust_remote_code=True,
        **({"token": args.hf_token} if args.hf_token else {}),
    )

    # Detect model architecture and select appropriate model class
    model_type = config.model_type.lower()
    if "mistral" in model_type:
        ModelClass = GearMistralForCausalLMNew
        model_arch_name = "Mistral"
    elif "llama" in model_type:
        ModelClass = GearLlamaForCausalLMNew
        model_arch_name = "Llama"
    else:
        raise ValueError(
            f"Unsupported model architecture: {model_type}. "
            f"Supported architectures: llama, mistral"
        )

    logging.info(f"Loading TrueCompression {model_arch_name} model from: {args.model}")
    model = ModelClass.from_pretrained(
        args.model,
        config=config,
        compress_config=compress_config,
        **model_kwargs,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    _p0 = next(model.parameters())
    logging.info(
        "Model parameters on device: %s (sample dtype=%s)",
        _p0.device,
        _p0.dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left",
        model_max_length=args.model_max_length,
        use_fast=False,
        cache_dir="../cache",
        **({"token": args.hf_token} if args.hf_token else {}),
    )
    tokenizer.pad_token = tokenizer.eos_token

    with open(args.prompt_file, "r") as f:
        prompt_prefix = f.read()

    all_samples = []
    total_acc = 0
    for batch in tqdm(dataloader, desc=f"Evaluate {args.dataset_split}"):
        questions = batch["question"]
        prompts, targets = prepare_aqua_cot_prompts_and_targets(batch, prompt_prefix)
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        inputs = inputs.to(device)
        generate_kwargs = dict(
            return_dict_in_generate=True,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            # repetition_penalty=1.3,
        )
        if args.do_sample:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = args.temperature
            generate_kwargs["top_k"] = args.top_k
            generate_kwargs["top_p"] = args.top_p
        else:
            generate_kwargs["do_sample"] = False
            generate_kwargs["temperature"] = None
            generate_kwargs["top_k"] = None
            generate_kwargs["top_p"] = None

        outputs = model.generate(**inputs, **generate_kwargs)
        generations = tokenizer.batch_decode(
            outputs.sequences[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        for question, generation, target in zip(prompts, generations, targets):
            pred, label, is_pred_true = evaluate_aqua_answer_cot(generation, target, question)
            if is_pred_true:
                total_acc += 1
            sample = EvaluationSample(
                question=question,
                generation=generation,
                target=target,
                pred=pred,
                label=label,
                is_pred_true=is_pred_true,
            )
            all_samples.append(sample)

    total_acc = total_acc / len(eval_dataset)
    evaluation_metric = EvaluationMetrics(accuracy=total_acc)
    evaluation_result = EvaluationResults(
        samples=all_samples,
        metrics=evaluation_metric,
    )

    logging.info('Evaluate %s acc: %.4f' % (args.dataset_split, total_acc))
    tb_writter.add_scalar(f"{args.dataset_split}/accuracy", total_acc, 1)
    with results_file.open("w") as handle:
        json.dump(evaluation_result.to_dict(), handle)

    # Get rank distribution data
    if get_rank_distribution is not None:
        rank_distribution = get_rank_distribution()
    else:
        rank_distribution = []
    
    if len(rank_distribution) > 0:
        logging.info("Recorded %d adaptive rank values", len(rank_distribution))
        if save_rank_distribution_to_csv is not None:
            csv_path = output_dir / "adaptive_rank_distribution.csv"
            save_rank_distribution_to_csv(str(csv_path))
            logging.info("Saved adaptive rank distribution CSV to %s", csv_path)
        if plt is not None:
            ranks = np.asarray(rank_distribution, dtype=np.int64)
            counts = np.bincount(ranks)
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.bar(np.arange(len(counts)), counts, color="skyblue", edgecolor='black', linewidth=0.5)
            ax.set_title("Adaptive Rank Distribution", fontsize=14, fontweight='bold')
            ax.set_xlabel("Rank", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            
            # Limit the number of x-ticks to avoid overcrowding
            num_ticks = min(len(counts), 20)  # Show at most 20 ticks
            tick_positions = np.linspace(0, len(counts) - 1, num_ticks, dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_positions, rotation=45, ha='right', fontsize=10)
            
            # Set minor ticks for all positions
            ax.set_xticks(np.arange(len(counts)), minor=True)
            ax.grid(True, which='minor', axis='x', alpha=0.2, linestyle=':')
            ax.grid(True, which='major', axis='y', alpha=0.3)
            
            fig.subplots_adjust(bottom=0.15)
            plot_path = output_dir / "rank_distribution.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            logging.info("Saved adaptive rank distribution plot to %s", plot_path)
        else:
            logging.warning("matplotlib not available; rank distribution plot was not generated")
    else:
        logging.info("No adaptive rank values were recorded; skipping rank distribution plot")
