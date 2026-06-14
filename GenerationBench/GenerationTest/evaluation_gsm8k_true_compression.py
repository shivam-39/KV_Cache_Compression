"""
GSM8K evaluation for the TrueCompression method (GearLlamaForCausalLMNew / GearMistralForCausalLMNew).

NOTE: TrueCompression supports Llama and Mistral-architecture models.
      Pass a local or HF Llama or Mistral checkpoint with --model.
      The script automatically detects the model architecture.

Compared to evaluation_gsm8k.py this script:
  - Imports GearLlamaForCausalLMNew for Llama or GearMistralForCausalLMNew for Mistral
  - Builds compress_config as a plain dict (no copy_for_all_attention / calculate_compress_ratio_list)
  - Adds --stream, --buffer_len, --compress_mode args for the new cache path
  - Defaults --example_subset to "0:10" and --model to TinyLlama for a cheap MPS smoke test
  - Auto-selects device: cuda → mps → cpu
  - Auto-detects model architecture (Llama or Mistral) and uses appropriate model class
  - Writes output JSON to evaluation_gsm8k_true_compression.json to avoid clobbering Simulated results
"""

import argparse
import json
import os
import logging
import re
import sys
import torch
import numpy as np
import datasets
import accelerate
import transformers

from tqdm.auto import tqdm
from pathlib import Path
from datasets import load_dataset
from typing import Any, Callable, Dict, Sequence, cast
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from torch.utils.tensorboard import SummaryWriter
from GEARLM import GearLlamaForCausalLMNew
from GEARLM.TrueCompression.models.TrueCompressionMistral import GearMistralForCausalLMNew
from transformers import AutoTokenizer

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    logging.warning("matplotlib not available; rank distribution plot will be skipped.")


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

MODEL_GENERATION_SPLIT = "\nQuestion: "
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationSample:
    """Wrapper around format evaluation sample."""

    question: str
    generation: str
    answer: str
    list_from_pred: list[str]
    list_from_answer: list[str]
    pred: float
    label: float
    is_pred_true: bool


@dataclass(frozen=True)
class EvaluationMetrics(DataClassJsonMixin):
    """Wrapper around aggregated evaluation metrics."""

    accuracy: float


@dataclass(frozen=True)
class EvaluationResults(DataClassJsonMixin):
    """Wrapper around evaluation results"""

    samples: list[EvaluationSample]
    metrics: EvaluationMetrics


def evaluate_pred_answer(pred_str, ans_str):
    pattern = r"\d*\.?\d+"
    pred_str, ans_str = pred_str.replace(",", ""), ans_str.replace(",", "")
    pred_list = re.findall(pattern, pred_str)
    gold_list = re.findall(pattern, ans_str)
    if len(pred_list) >= 1:
        pred = float(pred_list[-1])
        gold = float(gold_list[-1])
        is_pred_true = pred == gold
    else:
        is_pred_true = False
        pred = None
        gold = float(gold_list[-1])
    return (
        is_pred_true,
        pred,
        pred_list,
        gold,
        gold_list,
    )


def test_answer(pred_str, ans_str):
    pattern = r"\d*\.?\d+"
    pred = re.findall(pattern, pred_str)
    if len(pred) >= 1:
        print("#####\n Pred string:", pred_str, "\n pred_list", pred)
        pred = float(pred[-1].replace(",", ""))
        gold = re.findall(pattern, ans_str)
        print("\n Gold_answer", ans_str, "\n gold_list", gold)
        gold = float(gold[-1].replace(",", ""))
        print("\n result", gold, pred, gold == pred)
        return pred == gold
    else:
        return False


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model,
):
    """Resize tokenizer and embedding. (from original eval script)"""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GSM8K – TrueCompression")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Path or HF hub name of a Llama-architecture checkpoint. "
             "Defaults to TinyLlama for MPS smoke tests; swap for meta-llama/Llama-2-7b-hf on CUDA.",
    )
    parser.add_argument(
        "--prompt_file", type=str, default="gsm8k_prompt_original.txt", help=""
    )
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    # default 0:10 for a quick 10-example sanity run
    parser.add_argument(
        "--example_subset",
        type=str,
        default=None,
        help="Dataset slice, e.g. '0:10' or None for full test set.",
    )
    parser.add_argument("--max_length", type=int, default=None, help="")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="")
    parser.add_argument("--model_max_length", type=int, default=4096, help="")
    parser.add_argument("--do_sample", action="store_true", default=False, help="")
    parser.add_argument("--temperature", type=float, default=0.8, help="")
    parser.add_argument("--top_k", type=int, default=50, help="")
    parser.add_argument("--top_p", type=float, default=0.95, help="")
    parser.add_argument(
        "--generation_split", type=str, default=MODEL_GENERATION_SPLIT, help=""
    )
    parser.add_argument(
        "--root_output_dir", type=str, default="outputs", help="Root output dir"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--zero_shot", action="store_true", default=False, help="")
    parser.add_argument("--debug", action="store_true", default=False, help="")

    # ── TrueCompression-specific args ───────────────────────────────────────────
    parser.add_argument(
        "--compress_method",
        type=str,
        default="None",
        help="'None' to disable compression, or e.g. 'gear', 'uniform', 'outlier'.",
    )
    parser.add_argument(
        "--compress_mode",
        type=str,
        default="gear",
        help="Compress mode passed to CompressedUnion (gear / uniform / outlier / …).",
    )
    parser.add_argument("--quantize_bit", type=int, default=8, help="")
    parser.add_argument("--rank", type=float, default=0.0, help="")
    parser.add_argument("--loop", type=int, default=0, help="")
    parser.add_argument("--left", type=float, default=0.0, help="")
    parser.add_argument(
        "--buffer_len",
        type=int,
        default=20,
        help="Tail buffer length before compress() flushes (for new method).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Use StreamCompressedCache instead of CompressedCache.",
    )
    parser.add_argument(
        "--streaming_gap",
        type=int,
        default=1,
        help="Recompress interval for streaming cache.",
    )
    parser.add_argument("--sink_tokens", type=int, default=4,
    help="Number of initial prefill tokens always kept in full precision.")
    parser.add_argument("--recency_tokens", type=int, default=64,
    help="Size of the rolling recency FP window.")
    # ────────────────────────────────────────────────────────────────────────────

    args = parser.parse_args()

    if args.debug:
        import ipdb
        ipdb.set_trace()

    # ── Output paths ────────────────────────────────────────────────────────────
    root_output_dir = Path(args.root_output_dir)
    output_dir = f"cot_{args.prompt_file.split('.')[0]}_true_compression"
    if args.example_subset is not None:
        output_dir += f"_subset-{args.example_subset}"
    output_dir = root_output_dir / f"{args.model.split('/')[-1]}" / output_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    generation_file = output_dir / f"generation_results_subset-{args.example_subset}.txt"
    evaluation_result_file = output_dir / "evaluation_gsm8k_true_compression.json"

    # ── Dataset ─────────────────────────────────────────────────────────────────
    split = "test" if args.example_subset is None else f"test[{args.example_subset}]"
    eval_dataset = load_dataset("gsm8k", "main", split=split)

    # ── Logging ─────────────────────────────────────────────────────────────────
    tb_writter = SummaryWriter(log_dir=str(output_dir.resolve()))
    logging.basicConfig(
        filename=os.path.join(output_dir.resolve(), "log.txt"),
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # ── Device selection: cuda > mps > cpu ──────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # ── Build compress_config dict (for new method) ──────────────────────────────
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

    # ── Rank distribution tracking ──────────────────────────────────────────────
    try:
        from GEARLM.TrueCompression.models.rank_tracker import (
            clear_rank_distribution,
            get_rank_distribution,
            save_rank_distribution_to_csv,
        )
        clear_rank_distribution()
        logging.info("Initialized rank distribution tracking")
    except Exception as exc:
        logging.warning(
            "Could not initialize rank distribution tracking: %s",
            exc,
        )
        get_rank_distribution = None
        save_rank_distribution_to_csv = None

    # ── Model + tokenizer ────────────────────────────────────────────────────────
    # MPS does not support device_map="auto"; load to CPU first then move to device.
    # On CUDA, device_map="auto" is used for multi-GPU sharding if needed.
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
        use_flash_attn=True,
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

    # logging.info("Preprocessing the dataset.")
    # with open(f"lib_prompt/{args.prompt_file}", "r") as handle:
    #     prompt_cot = handle.read()
    logging.info("Preprocessing the dataset.")
    with open(f"lib_prompt/{args.prompt_file}", "r") as handle:
            prompt_cot = handle.read()
    

    # few_shot_pairs = []
    # if not args.zero_shot:
    #     blocks = [b.strip() for b in prompt_cot.strip().split("\n\n") if b.strip()]
    #     for block in blocks:
    #         lines = block.split("\n")
    #         ex_q = lines[0].replace("Question: ", "", 1).strip()
    #         ex_a = "\n".join(lines[1:]).strip()
    #         few_shot_pairs.append((ex_q, ex_a))
    logging.info(f"Loaded prompt from lib_prompt/{args.prompt_file}")
    dataloader = torch.utils.data.DataLoader(
        cast(torch.utils.data.Dataset, eval_dataset),
        batch_size=args.batch_size,
    )

    # ── Eval loop ────────────────────────────────────────────────────────────────
    all_samples = []
    all_question, all_generation, all_answer = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate GSM8K (TrueCompression)"):
            questions = batch["question"]
            answers = batch["answer"]

            # if args.zero_shot:
            #     prompt_cot = (
            #         "answer the question through the form of The answer is xxx."
            #         " Do not generate others."
            #     )
            # prompts = [
            #     prompt_cot + "\nQuestion: " + question + "\n"
            #     for question in questions
            # ]
            # template of the tinyllama prompt :
            # prompts = []
            # for question in questions:
            #     messages = [
            #         {
            #             "role": "system",
            #             "content": "You are a helpful math assistant. Answer step by step."
            #         },
            #         {
            #             "role": "user",
            #             "content": prompt_cot + "\nQuestion: " + question + "\n"
            #         },
            #     ]
            #     prompts.append(
            #         tokenizer.apply_chat_template(
            #             messages,
            #             tokenize=False,
            #             add_generation_prompt=True,
            #         )
            #     )

            # prompts = []
            # for question in questions:
            #     if args.zero_shot:
            #         messages = [
            #             {
            #                 "role": "system",
            #                 "content": "Solve math problems. Respond only with: The answer is X.",
            #             },
            #             {"role": "user", "content": question},
            #         ]
            #     else:
            #         # Each few-shot example is a separate user/assistant turn so the
            #         # model sees the CoT format as prior assistant behaviour to imitate.
            #         messages = [
            #             {
            #                 "role": "system",
            #                 "content": (
            #                     "You are a helpful math assistant. "
            #                     "Solve problems step by step and end every answer with '#### <number>'."
            #                 ),
            #             }
            #         ]
            #         for ex_q, ex_a in few_shot_pairs:
            #             messages.append({"role": "user", "content": ex_q})
            #             messages.append({"role": "assistant", "content": ex_a})
            #         messages.append({"role": "user", "content": question})

            #     prompts.append(
            #         tokenizer.apply_chat_template(
            #             messages,
            #             tokenize=False,
            #             add_generation_prompt=True,
            #         )
            #     )
            if args.zero_shot:
                prompts = [
                    "Answer the question. The answer is xxx.\nQuestion: " + question + "\n"
                    for question in questions
                ]
            else:
                prompts = [
                    prompt_cot + "\nQuestion: " + question + "\n"
                    for question in questions
                ]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
            )
            print(inputs.input_ids.shape)
            inputs = inputs.to(device)
             #cosmetic changes for fixing the generation 
            generate_kwargs = dict(
                return_dict_in_generate=True,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            ) 
            # generate_kwargs = dict(
            #     return_dict_in_generate=True,
            #     max_length=args.max_length,
            #     max_new_tokens=args.max_new_tokens,
            #     pad_token_id=tokenizer.eos_token_id,
            #     use_cache=True,
            #     # repetition_penalty=1.3, #to avoid the repetition of the same answer
            #     # stop_sequences=["####"], #the token to stop the generation
            #     # stopping_criteria=[StoppingCriteriaSub(stops=[tokenizer.encode("####")])],
            # )
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

            all_question += questions
            all_generation += generations
            all_answer += answers

            for question, generation, answer in zip(questions, generations, answers):
                is_pred_true, pred, pred_list, gold, gold_list = evaluate_pred_answer(
                    generation.split(args.generation_split)[0], answer
                )
                sample = EvaluationSample(
                    question=question,
                    generation=generation,
                    answer=answer,
                    list_from_pred=pred_list,
                    list_from_answer=gold_list,
                    pred=pred,
                    label=gold,
                    is_pred_true=is_pred_true,
                )
                all_samples.append(sample)

        accuracy = sum([sample.is_pred_true for sample in all_samples]) / len(all_samples)
        evaluation_metric = EvaluationMetrics(accuracy=accuracy)
        evaluation_result = EvaluationResults(
            samples=all_samples,
            metrics=evaluation_metric,
        )

    tb_writter.add_scalar("accuracy", accuracy, 1)
    logging.info(f"Accuracy: {accuracy}")

    with evaluation_result_file.open("w") as handle:
        json.dump(evaluation_result.to_dict(), handle)

    # ── Rank distribution outputs ────────────────────────────────────────────────
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
            ax.bar(np.arange(len(counts)), counts, color="skyblue", edgecolor="black", linewidth=0.5)
            ax.set_title("Adaptive Rank Distribution", fontsize=14, fontweight="bold")
            ax.set_xlabel("Rank", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)

            num_ticks = min(len(counts), 20)
            tick_positions = np.linspace(0, len(counts) - 1, num_ticks, dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_positions, rotation=45, ha="right", fontsize=10)

            ax.set_xticks(np.arange(len(counts)), minor=True)
            ax.grid(True, which="minor", axis="x", alpha=0.2, linestyle=":")
            ax.grid(True, which="major", axis="y", alpha=0.3)

            fig.subplots_adjust(bottom=0.15)
            plot_path = output_dir / "rank_distribution.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            logging.info("Saved adaptive rank distribution plot to %s", plot_path)
        else:
            logging.warning("matplotlib not available; rank distribution plot was not generated")
    else:
        logging.info("No adaptive rank values were recorded; skipping rank distribution plot")

    with generation_file.open("w", encoding="utf-8") as handle:
        for question, generation, answer in zip(
            all_question, all_generation, all_answer
        ):
            handle.write("Q: %s\nA_model:\n%s\nA:\n%s\n\n" % (question, generation, answer))

    print(f"\nAccuracy on {len(all_samples)} examples: {accuracy:.4f}")
    print(f"Results saved to: {evaluation_result_file}")
