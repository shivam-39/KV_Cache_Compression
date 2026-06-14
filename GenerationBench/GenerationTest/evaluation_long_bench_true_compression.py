"""
LongBench evaluation for TrueCompression (GearLlamaForCausalLMNew).

Uses official LongBench prompt templates and task-specific metrics.

NOTE: TrueCompression currently only supports Llama-architecture models.

Usage (smoke test):
    cd GenerationBench/GenerationTest

    python long_bench_test.py \\
        --long_bench_subset narrativeqa \\
        --example_subset 0:2 \\
        --compress_method None

    python long_bench_test.py \\
        --long_bench_subset gov_report \\
        --example_subset 0:2 \\
        --compress_method None
"""

import argparse
import json
import logging
import os
import re
import string
import sys
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, cast

import torch
import transformers
from dataclasses_json import DataClassJsonMixin
from datasets import load_dataset
from rouge_score import rouge_scorer
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from GEARLM import GearLlamaForCausalLMNew


LONG_BENCH_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
]

# Official LongBench prompt templates (THUDM/LongBench/config/dataset2prompt.json)
DATASET2PROMPT = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question asconcisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\nStory: {context}\n\n"
        "Now, answer the question based on the story asconcisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be answered based on the "
        "information in the article, write \"unanswerable\". If the question is a yes/no question, answer "
        "\"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, using a single phrase or "
        "sentence if possible. If the question cannot be answered based on the information in the article, "
        "write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or "
        "\"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer and do not "
        "output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
   
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\n"
        "Now, write a one-page summary of all the news.\n\nSummary:"
    ),
   
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not output any other "
        "words. The following are some examples.\n\n{context}\n\n{input}"
    ),
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
   
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully "
        "read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In "
        "other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. The output format should "
        "only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the "
        "abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. The answer format must be like "
        "\"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: "
    ),
   
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

# Official default max generation lengths (THUDM/LongBench/config/dataset2maxlen.json)
DATASET2MAXLEN = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}

ROUGE_TASKS = {"gov_report", "qmsum", "multi_news", "samsum"}
FIRST_LINE_TASKS = {"trec", "triviaqa", "samsum"}

ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


@dataclass(frozen=True)
class EvaluationSample:
    input_text: str
    context: str
    generation: str
    references: list[str]
    score: float
    length: int


@dataclass(frozen=True)
class EvaluationMetrics(DataClassJsonMixin):
    task: str
    metric_name: str
    score: float  # LongBench reports 0-100


@dataclass(frozen=True)
class EvaluationResults(DataClassJsonMixin):
    samples: list[EvaluationSample]
    metrics: EvaluationMetrics


# ── LongBench metrics (adapted from THUDM/LongBench/metrics.py) ───────────────

def normalize_answer(text: str) -> str:
    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punc(s):
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    text = text.lower()
    text = remove_punc(text)
    text = remove_articles(text)
    return " ".join(text.split())


def token_f1(prediction_tokens, ground_truth_tokens) -> float:
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    return token_f1(pred_tokens, gold_tokens)


def qa_f1_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    try:
        import jieba
    except ImportError as exc:
        raise ImportError("Install jieba for Chinese LongBench tasks: pip install jieba") from exc

    def normalize_zh_answer(s: str) -> str:
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        s = "".join(ch for ch in s if ch not in all_punctuation)
        return "".join(s.split()).lower()

    pred_tokens = [normalize_zh_answer(t) for t in jieba.cut(prediction, cut_all=False)]
    gold_tokens = [normalize_zh_answer(t) for t in jieba.cut(ground_truth, cut_all=False)]
    pred_tokens = [t for t in pred_tokens if t]
    gold_tokens = [t for t in gold_tokens if t]
    return token_f1(pred_tokens, gold_tokens)


def rouge_l_score(prediction: str, ground_truth: str, **kwargs) -> float:
    scores = ROUGE_SCORER.score(ground_truth, prediction)
    return scores["rougeL"].fmeasure


def rouge_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    try:
        import jieba
    except ImportError as exc:
        raise ImportError("Install jieba for Chinese LongBench tasks: pip install jieba") from exc
    pred = " ".join(jieba.cut(prediction, cut_all=False))
    gold = " ".join(jieba.cut(ground_truth, cut_all=False))
    return rouge_l_score(pred, gold)


def classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
    all_classes = kwargs.get("all_classes") or []
    em_match_list = [c for c in all_classes if c in prediction]
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def retrieval_score(prediction: str, ground_truth: str, **kwargs) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return right_num / len(numbers)


def retrieval_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    matches = re.findall(r"段落(\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return right_num / len(numbers)


def count_score(prediction: str, ground_truth: str, **kwargs) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth))
    return right_num / len(numbers)


def code_sim_score(prediction: str, ground_truth: str, **kwargs) -> float:
    pred_line = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            pred_line = line
            break
    return SequenceMatcher(None, pred_line, ground_truth).ratio()


DATASET2METRIC: dict[str, Callable[..., float]] = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_l_score,
    "qmsum": rouge_l_score,
    "multi_news": rouge_l_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_l_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

METRIC_NAMES = {
    "narrativeqa": "qa_f1",
    "qasper": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "multifieldqa_zh": "qa_f1_zh",
    "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1",
    "musique": "qa_f1",
    "dureader": "rouge_l_zh",
    "gov_report": "rouge_l",
    "qmsum": "rouge_l",
    "multi_news": "rouge_l",
    "vcsum": "rouge_l_zh",
    "trec": "classification",
    "triviaqa": "qa_f1",
    "samsum": "rouge_l",
    "lsht": "classification",
    "passage_retrieval_en": "retrieval",
    "passage_count": "count",
    "passage_retrieval_zh": "retrieval_zh",
    "lcc": "code_sim",
    "repobench-p": "code_sim",
}


def build_longbench_prompt(task: str, context: str, input_text: str) -> str:
    template = DATASET2PROMPT[task]
    if "{input}" in template:
        return template.format(context=context, input=input_text)
    return template.format(context=context)


def postprocess_prediction(task: str, prediction: str) -> str:
    if task in FIRST_LINE_TASKS:
        return prediction.lstrip("\n").split("\n")[0]
    return prediction.strip()


def score_prediction(
    task: str,
    prediction: str,
    answers: list[str],
    all_classes: Any = None,
) -> float:
    prediction = postprocess_prediction(task, prediction)
    metric_fn = DATASET2METRIC[task]
    score = 0.0
    for ground_truth in answers:
        kwargs = {}
        if all_classes is not None:
            kwargs["all_classes"] = all_classes
        score = max(score, metric_fn(prediction, ground_truth, **kwargs))
    return score


def truncate_prompt_middle(tokenizer, prompt: str, max_length: int) -> str:
    token_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(token_ids) <= max_length:
        return prompt
    half = max_length // 2
    return (
        tokenizer.decode(token_ids[:half], skip_special_tokens=True)
        + tokenizer.decode(token_ids[-half:], skip_special_tokens=True)
    )


def build_generate_kwargs(args, tokenizer, max_new_tokens: int):
    generate_kwargs = {
        "return_dict_in_generate": True,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": args.do_sample,
    }
    if args.max_length is not None:
        generate_kwargs["max_length"] = args.max_length
    if args.do_sample:
        generate_kwargs.update(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    return generate_kwargs


def wrap_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LongBench – TrueCompression")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Path or HF hub name of a Llama-architecture checkpoint.",
    )
    parser.add_argument(
        "--long_bench_subset",
        type=str,
        default="narrativeqa",
        choices=LONG_BENCH_TASKS,
        help="Which LongBench task to evaluate.",
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
        default=None,
        help="Max tokens to generate. Defaults to official LongBench value per task.",
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=8192,
        help="Max input tokens after middle truncation.",
    )
    parser.add_argument("--do_sample", action="store_true", default=False, help="Enable sampling.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty during generation.",
    )
    parser.add_argument(
        "--root_output_dir",
        type=str,
        default="outputs",
        help="Root output directory.",
    )
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        default=False,
        help="Wrap LongBench plain-text prompt with tokenizer chat template.",
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
        help="Compress mode passed to CompressedUnion.",
    )
    parser.add_argument("--quantize_bit", type=int, default=8, help="Quantization bits.")
    parser.add_argument("--rank", type=float, default=0.0, help="Low-rank factor.")
    parser.add_argument("--loop", type=int, default=0, help="Power iteration loops.")
    parser.add_argument("--left", type=float, default=0.0, help="Outlier fraction.")
    parser.add_argument("--buffer_len", type=int, default=20, help="FP buffer length.")
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Use StreamCompressedCache instead of CompressedCache.",
    )
    parser.add_argument("--streaming_gap", type=int, default=1, help="Streaming recompress gap.")
    parser.add_argument("--sink_tokens", type=int, default=4, help="Sink tokens in FP.")
    parser.add_argument("--recency_tokens", type=int, default=64, help="Recency FP window size.")
    # ────────────────────────────────────────────────────────────────────────────

    args = parser.parse_args()

    if args.example_subset.lower() == "none":
        args.example_subset = None

    if args.debug:
        import ipdb
        ipdb.set_trace()

    task = args.long_bench_subset
    max_gen = args.max_new_tokens or DATASET2MAXLEN[task]

    # ── Output paths ────────────────────────────────────────────────────────────
    root_output_dir = Path(args.root_output_dir)
    output_dir = f"longbench_{task}_true_compression"
    if args.example_subset is not None:
        output_dir += f"_subset-{args.example_subset}"
    output_dir = root_output_dir / args.model.split("/")[-1] / output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    subset_label = args.example_subset if args.example_subset is not None else "full"
    generation_file = output_dir / f"generations_subset-{subset_label}.txt"
    evaluation_result_file = output_dir / f"evaluation_longbench_{task}.json"

    # ── Dataset ─────────────────────────────────────────────────────────────────
    split = "test" if args.example_subset is None else f"test[{args.example_subset}]"
    # eval_dataset = load_dataset("THUDM/LongBench", task, split=split)
    eval_dataset = load_dataset("THUDM/LongBench", task, split=split)
    if "all_classes" in eval_dataset.column_names:
        eval_dataset = eval_dataset.remove_columns(["all_classes"])

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
    logging.info("LongBench task: %s", task)
    logging.info("Dataset split: %s", split)
    logging.info("Metric: %s", METRIC_NAMES[task])
    logging.info("max_new_tokens: %d", max_gen)

    # ── Device ──────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info("Using device: %s", device)

    # ── compress_config ─────────────────────────────────────────────────────────
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

    # ── Model + tokenizer ───────────────────────────────────────────────────────
    hf_token_kwargs = {"token": args.hf_token} if args.hf_token else {}
    if device.type == "cuda":
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "cache_dir": "../cache",
            **hf_token_kwargs,
        }
    else:
        model_kwargs = {
            "torch_dtype": torch.float16,
            "cache_dir": "../cache",
            **hf_token_kwargs,
        }

    config = transformers.AutoConfig.from_pretrained(
        args.model,
        use_flash_attn=False,
        trust_remote_code=True,
        **hf_token_kwargs,
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
        **hf_token_kwargs,
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataloader = torch.utils.data.DataLoader(
        cast(torch.utils.data.Dataset, eval_dataset),
        batch_size=args.batch_size,
    )

    # ── Eval loop ───────────────────────────────────────────────────────────────
    all_samples = []
    all_inputs, all_contexts, all_generations, all_references = [], [], [], []
    per_example_scores = []

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc=f"Evaluate LongBench/{task} (TrueCompression)",
        ):
            input_texts = batch["input"]
            contexts = batch["context"]
            answers_batch = batch["answers"]
            lengths = batch["length"]
            all_classes_batch = batch.get("all_classes")

            prompts = []
            for context, input_text in zip(contexts, input_texts):
                prompt = build_longbench_prompt(task, context, input_text)
                prompt = truncate_prompt_middle(tokenizer, prompt, args.model_max_length)
                if args.use_chat_template:
                    prompt = wrap_chat_prompt(tokenizer, prompt)
                prompts.append(prompt)

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                truncation=False,
            )
            logging.info("Input token shape: %s", tuple(inputs.input_ids.shape))
            inputs = inputs.to(device)

            generate_kwargs = build_generate_kwargs(args, tokenizer, max_gen)
            outputs = model.generate(**inputs, **generate_kwargs)
            generations = tokenizer.batch_decode(
                outputs.sequences[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            for i, (input_text, context, generation, answers, length) in enumerate(
                zip(input_texts, contexts, generations, answers_batch, lengths)
            ):
                all_classes = None
                if all_classes_batch is not None:
                    all_classes = all_classes_batch[i]
                    if all_classes is None or all_classes == []:
                        all_classes = None

                score = score_prediction(task, generation, answers, all_classes=all_classes)
                per_example_scores.append(score)

                all_inputs.append(input_text)
                all_contexts.append(context)
                all_generations.append(generation)
                all_references.append(answers)

                all_samples.append(
                    EvaluationSample(
                        input_text=input_text,
                        context=context,
                        generation=generation,
                        references=answers,
                        score=round(100 * score, 4),
                        length=int(length),
                    )
                )

    final_score = round(100 * sum(per_example_scores) / max(len(per_example_scores), 1), 2)
    evaluation_metric = EvaluationMetrics(
        task=task,
        metric_name=METRIC_NAMES[task],
        score=final_score,
    )
    evaluation_result = EvaluationResults(
        samples=all_samples,
        metrics=evaluation_metric,
    )

    tb_writter.add_scalar("longbench_score", final_score, 1)

    logging.info(
        "LongBench/%s (%s): %.2f",
        task,
        METRIC_NAMES[task],
        final_score,
    )

    with evaluation_result_file.open("w", encoding="utf-8") as handle:
        json.dump(evaluation_result.to_dict(), handle, indent=2, ensure_ascii=False)

    with generation_file.open("w", encoding="utf-8") as handle:
        for input_text, context, generation, references in zip(
            all_inputs, all_contexts, all_generations, all_references
        ):
            handle.write(
                "Input:\n%s\n\nContext:\n%s\n\nModel:\n%s\n\nReferences:\n%s\n\n"
                % (input_text, context, generation, references)
            )

    print(f"\nLongBench/{task} ({METRIC_NAMES[task]}): {final_score:.2f}")
    print(f"Results saved to: {evaluation_result_file}")
    print(f"Generations saved to: {generation_file}")