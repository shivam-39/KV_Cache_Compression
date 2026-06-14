"""
SCROLLS evaluation for TrueCompression (GearLlamaForCausalLMNew).

Generates task outputs on tau/scrolls subsets and scores with ROUGE.

NOTE: TrueCompression currently only supports Llama-architecture models.
      Pass a local or HF Llama checkpoint with --model.

Usage (smoke test):
    python scrolls_test.py \\
        --scrolls_subset gov_report \\
        --example_subset 0:2 \\
        --zero_shot \\
        --max_new_tokens 512 \\
        --compress_method None
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import transformers
from dataclasses_json import DataClassJsonMixin
from datasets import load_dataset
from rouge_score import rouge_scorer
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from GEARLM import GearLlamaForCausalLMNew


SCROLLS_SUBSETS = [
    "gov_report",
    "summ_screen_fd",
    "qmsum",
    "narrative_qa",
    "qasper",
    "quality",
    "contract_nli",
]


@dataclass(frozen=True)
class EvaluationSample:
    input_text: str
    generation: str
    reference: str
    rouge1: float
    rouge2: float
    rougeL: float


@dataclass(frozen=True)
class EvaluationMetrics(DataClassJsonMixin):
    rouge1: float
    rouge2: float
    rougeL: float


@dataclass(frozen=True)
class EvaluationResults(DataClassJsonMixin):
    samples: list[EvaluationSample]
    metrics: EvaluationMetrics


def mean_rouge(scores_list):
    if not scores_list:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    keys = ["rouge1", "rouge2", "rougeL"]
    return {
        k: sum(s[k].fmeasure for s in scores_list) / len(scores_list)
        for k in keys
    }


def parse_few_shot_pairs(prompt_text: str) -> list[tuple[str, str]]:
    """
    Parse few-shot examples from a prompt file.

    Expected block format:
        Document:
        <document text>

        Summary:
        <summary text>
    """
    pairs = []
    blocks = [b.strip() for b in prompt_text.strip().split("\n\n") if b.strip()]
    for block in blocks:
        if "Summary:" not in block:
            continue
        doc_part, sum_part = block.split("Summary:", 1)
        ex_doc = doc_part.replace("Document:", "", 1).strip()
        ex_sum = sum_part.strip()
        if ex_doc and ex_sum:
            pairs.append((ex_doc, ex_sum))
    return pairs


def build_messages(document: str, zero_shot: bool, few_shot_pairs: list[tuple[str, str]]):
    if zero_shot:
        return [
            {
                "role": "system",
                "content": "Summarise the document in 2 paragraphs.",
            },
            {"role": "user", "content": document},
        ]

    messages = [{"role": "system", "content": "You are an informative summariser."}]
    for ex_doc, ex_sum in few_shot_pairs:
        messages.append({"role": "user", "content": ex_doc})
        messages.append({"role": "assistant", "content": ex_sum})
    messages.append({"role": "user", "content": document})
    return messages


def build_generate_kwargs(args, tokenizer):
    generate_kwargs = {
        "return_dict_in_generate": True,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.max_length is not None:
        generate_kwargs["max_length"] = args.max_length

    if args.do_sample:
        generate_kwargs.update(
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    else:
        generate_kwargs.update(
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
        )
    return generate_kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SCROLLS – TrueCompression")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Path or HF hub name of a Llama-architecture checkpoint.",
    )
    parser.add_argument(
        "--scrolls_subset",
        type=str,
        default="gov_report",
        choices=SCROLLS_SUBSETS,
        help="Which SCROLLS task/subset to evaluate.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="scrolls_prompt_original.txt",
        help="Few-shot prompt file in lib_prompt/ (used when --zero_shot is not set).",
    )
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--example_subset",
        type=str,
        default="0:10",
        help="Dataset slice, e.g. '0:10'. Pass 'none' for full test set.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Optional total max length cap (input + output).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=5000,
        help="Max tokens to generate.",
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=10000,
        help="Tokenizer max length (inputs may still be truncated).",
    )
    parser.add_argument("--do_sample", action="store_true", default=False, help="Enable sampling.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.3,
        help="Repetition penalty during generation.",
    )
    parser.add_argument(
        "--root_output_dir",
        type=str,
        default="outputs",
        help="Root output directory.",
    )
    parser.add_argument(
        "--zero_shot",
        action="store_true",
        default=True,
        help="Use zero-shot summarisation prompt (default: True).",
    )
    parser.add_argument(
        "--few_shot",
        action="store_true",
        default=False,
        help="Use few-shot examples from --prompt_file instead of zero-shot.",
    )
    parser.add_argument("--debug", action="store_true", default=False, help="Drop into ipdb.")

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
    parser.add_argument("--quantize_bit", type=int, default=8, help="Quantization bits.")
    parser.add_argument("--rank", type=float, default=0.0, help="Low-rank factor.")
    parser.add_argument("--loop", type=int, default=0, help="Power iteration loops.")
    parser.add_argument("--left", type=float, default=0.0, help="Outlier fraction.")
    parser.add_argument(
        "--buffer_len",
        type=int,
        default=20,
        help="Tail buffer length before compress() flushes.",
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
    parser.add_argument(
        "--sink_tokens",
        type=int,
        default=4,
        help="Initial prefill tokens kept in full precision.",
    )
    parser.add_argument(
        "--recency_tokens",
        type=int,
        default=64,
        help="Size of the rolling recency FP window.",
    )
    # ────────────────────────────────────────────────────────────────────────────

    args = parser.parse_args()

    if args.few_shot:
        args.zero_shot = False

    if args.example_subset.lower() == "none":
        args.example_subset = None

    if args.debug:
        import ipdb
        ipdb.set_trace()

    # ── Output paths ────────────────────────────────────────────────────────────
    root_output_dir = Path(args.root_output_dir)
    output_dir = f"scrolls_{args.scrolls_subset}_true_compression"
    if args.example_subset is not None:
        output_dir += f"_subset-{args.example_subset}"
    output_dir = root_output_dir / args.model.split("/")[-1] / output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    subset_label = args.example_subset if args.example_subset is not None else "full"
    generation_file = output_dir / f"generations_subset-{subset_label}.txt"
    evaluation_result_file = output_dir / f"evaluation_scrolls_{args.scrolls_subset}.json"

    # ── Dataset ─────────────────────────────────────────────────────────────────
    split = "test" if args.example_subset is None else f"test[{args.example_subset}]"
    eval_dataset = load_dataset("tau/scrolls", args.scrolls_subset, split=split)

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

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
    logging.info("SCROLLS subset: %s", args.scrolls_subset)
    logging.info("Example split: %s", split)

    # ── Device selection: cuda > mps > cpu ──────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info("Using device: %s", device)

    # ── Build compress_config dict ───────────────────────────────────────────────
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
    logging.info("compress_config: %s", compress_config)

    # ── Model + tokenizer ────────────────────────────────────────────────────────
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

    logging.info("Loading TrueCompression Llama model from: %s", args.model)
    model = GearLlamaForCausalLMNew.from_pretrained(
        args.model,
        config=config,
        compress_config=compress_config,
        **model_kwargs,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left",
        model_max_length=args.model_max_length,
        use_fast=False,
        cache_dir="../cache",
        **({"token": args.hf_token} if args.hf_token else {}),
    )
    tokenizer.pad_token = tokenizer.eos_token

    few_shot_pairs = []
    if not args.zero_shot:
        prompt_path = Path("lib_prompt") / args.prompt_file
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Few-shot prompt file not found: {prompt_path}")
        prompt_cot = prompt_path.read_text(encoding="utf-8")
        few_shot_pairs = parse_few_shot_pairs(prompt_cot)
        logging.info("Parsed %d few-shot examples from %s", len(few_shot_pairs), prompt_path)
        if not few_shot_pairs:
            raise ValueError(
                f"No few-shot examples found in {prompt_path}. "
                "Expected blocks with 'Document:' and 'Summary:'."
            )

    dataloader = torch.utils.data.DataLoader(
        cast(torch.utils.data.Dataset, eval_dataset),
        batch_size=args.batch_size,
    )

    # ── Eval loop ────────────────────────────────────────────────────────────────
    all_samples = []
    all_documents, all_generations, all_references = [], [], []
    rouge_scores_list = []

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc=f"Evaluate SCROLLS/{args.scrolls_subset} (TrueCompression)",
        ):
            documents = batch["input"]
            references = batch["output"]

            prompts = []
            for document in documents:
                messages = build_messages(document, args.zero_shot, few_shot_pairs)
                prompts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
            )
            logging.info("Input token shape: %s", tuple(inputs.input_ids.shape))
            inputs = inputs.to(device)

            generate_kwargs = build_generate_kwargs(args, tokenizer)
            outputs = model.generate(**inputs, **generate_kwargs)
            generations = tokenizer.batch_decode(
                outputs.sequences[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            all_documents += documents
            all_generations += generations
            all_references += references

            for document, generation, reference in zip(documents, generations, references):
                scores = scorer.score(reference, generation)
                rouge_scores_list.append(scores)
                all_samples.append(
                    EvaluationSample(
                        input_text=document,
                        generation=generation,
                        reference=reference,
                        rouge1=scores["rouge1"].fmeasure,
                        rouge2=scores["rouge2"].fmeasure,
                        rougeL=scores["rougeL"].fmeasure,
                    )
                )

    metrics_dict = mean_rouge(rouge_scores_list)
    evaluation_metric = EvaluationMetrics(
        rouge1=metrics_dict["rouge1"],
        rouge2=metrics_dict["rouge2"],
        rougeL=metrics_dict["rougeL"],
    )
    evaluation_result = EvaluationResults(
        samples=all_samples,
        metrics=evaluation_metric,
    )

    tb_writter.add_scalar("rouge1", evaluation_metric.rouge1, 1)
    tb_writter.add_scalar("rouge2", evaluation_metric.rouge2, 1)
    tb_writter.add_scalar("rougeL", evaluation_metric.rougeL, 1)

    logging.info(
        "ROUGE — rouge1: %.4f, rouge2: %.4f, rougeL: %.4f",
        evaluation_metric.rouge1,
        evaluation_metric.rouge2,
        evaluation_metric.rougeL,
    )

    with evaluation_result_file.open("w", encoding="utf-8") as handle:
        json.dump(evaluation_result.to_dict(), handle, indent=2)

    with generation_file.open("w", encoding="utf-8") as handle:
        for document, generation, reference in zip(
            all_documents, all_generations, all_references
        ):
            handle.write(
                "Document:\n%s\n\nModel:\n%s\n\nReference:\n%s\n\n"
                % (document, generation, reference)
            )

    print(f"\nROUGE-L on {len(all_samples)} examples: {evaluation_metric.rougeL:.4f}")
    print(f"Results saved to: {evaluation_result_file}")
    print(f"Generations saved to: {generation_file}")

    