#!/usr/bin/env python3
"""Estimate GEAR KV-cache memory use from adaptive-rank CSV output.

The rank tracker records one adaptive rank for each compressed K or V matrix.
For a matrix reshaped to [seq_len, kv_dim], this script compares:

  original FP cache:      seq_len * kv_dim * original_bits
  compressed GEAR cache:  sink_recency * kv_dim * original_bits
                          + compressed_seq_len * kv_dim * quant_bits
                          + (compressed_seq_len + kv_dim + 1) * rank * residual_bits

where compressed_seq_len = seq_len - sink_recency.

The final percentage treats the original, uncompressed cache as 100%.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
# DEFAULT_CSV = SCRIPT_DIR / "outputs/Meta-Llama-3-8B/cot_gsm8k_prompt_original_true_compression_subset-0:10/adaptive_rank_distribution_5.csv"
# DEFAULT_CSV = SCRIPT_DIR / "outputs/Meta-Llama-3-8B/cot_aqua_cot_true_compression_subset-0:10/adaptive_rank_distribution_5.csv"
# DEFAULT_CSV = SCRIPT_DIR / "outputs/Mistral-7B-Instruct-v0.3/cot_gsm8k_prompt_original_true_compression_subset-0:10/adaptive_rank_distribution_5.csv"
DEFAULT_CSV = SCRIPT_DIR / "outputs/Mistral-7B-Instruct-v0.3/cot_aqua_cot_true_compression_subset-0:10/adaptive_rank_distribution_5.csv"


MODEL_PRESETS = {
    # Effective K/V width is num_key_value_heads * head_dim, not necessarily
    # hidden_size when grouped-query attention is used.
    "mistral-7b": 8 * 128,
    "llama3-8b": 8 * 128,
    "llama2-7b": 32 * 128,
    "llama2-13b": 40 * 128,
}


@dataclass(frozen=True)
class CompressionEstimate:
    rank_count: int
    seq_len: int
    sink_recency: int
    compressed_seq_len: int
    kv_dim: int
    quant_bits: int
    residual_bits: int
    original_bits: int
    original_total_bits: float
    compressed_total_bits: float

    @property
    def compressed_percent_of_original(self) -> float:
        return 100.0 * self.compressed_total_bits / self.original_total_bits

    @property
    def reduction_percent(self) -> float:
        return 100.0 - self.compressed_percent_of_original

    @property
    def compression_ratio(self) -> float:
        return self.original_total_bits / self.compressed_total_bits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate compressed KV-cache size percentage from "
            "adaptive_rank_distribution_5.csv."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Rank CSV path. Default: {DEFAULT_CSV}",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help=(
            "Sequence length of each compressed matrix. If omitted, the script "
            "uses max(adaptive_rank) as a lower-bound estimate."
        ),
    )
    parser.add_argument(
        "--model-dim",
        "--kv-dim",
        dest="model_dim",
        type=int,
        default=None,
        help="Effective K/V width: num_key_value_heads * head_dim.",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Alternative to --model-dim; used with --head-dim.",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=None,
        help="Alternative to --model-dim; used with --num-kv-heads.",
    )
    parser.add_argument(
        "--model-preset",
        choices=sorted(MODEL_PRESETS),
        default="mistral-7b",
        help=(
            "Preset for effective K/V width. Ignored when --model-dim or "
            "--num-kv-heads/--head-dim is given."
        ),
    )
    parser.add_argument(
        "--quant-bits",
        type=int,
        default=4,
        help="Fixed quantization bits for K and V matrices. Default: 4.",
    )
    parser.add_argument(
        "--original-bits",
        type=int,
        default=16,
        help="Bit width of the original uncompressed cache. Default: 16.",
    )
    parser.add_argument(
        "--residual-bits",
        type=int,
        default=16,
        help="Bit width for low-rank SVD factors U/S/V. Default: 16.",
    )
    parser.add_argument(
        "--sink-recency",
        type=int,
        default=8,
        help=(
            "Number of sink/recency tokens kept uncompressed in full precision. "
            "Default: 0."
        ),
    )
    parser.add_argument(
        "--per-row-output",
        type=Path,
        default=None,
        help="Optional CSV path for per-rank compressed percentage details.",
    )
    return parser.parse_args()


def read_adaptive_ranks(csv_path: Path) -> List[int]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Rank CSV not found: {csv_path}")

    ranks: List[int] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "adaptive_rank" not in (reader.fieldnames or []):
            raise ValueError(
                f"{csv_path} must contain an 'adaptive_rank' column; "
                f"found {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            raw_rank = row.get("adaptive_rank", "").strip()
            if not raw_rank:
                continue
            try:
                rank = int(float(raw_rank))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid adaptive_rank at {csv_path}:{row_number}: {raw_rank!r}"
                ) from exc
            if rank < 0:
                raise ValueError(
                    f"adaptive_rank must be non-negative at {csv_path}:{row_number}"
                )
            ranks.append(rank)

    if not ranks:
        raise ValueError(f"No adaptive ranks found in {csv_path}")
    return ranks


def resolve_kv_dim(args: argparse.Namespace) -> int:
    if args.model_dim is not None:
        return positive_int(args.model_dim, "--model-dim")

    if args.num_kv_heads is not None or args.head_dim is not None:
        if args.num_kv_heads is None or args.head_dim is None:
            raise ValueError("--num-kv-heads and --head-dim must be provided together")
        return positive_int(args.num_kv_heads, "--num-kv-heads") * positive_int(
            args.head_dim, "--head-dim"
        )

    return MODEL_PRESETS[args.model_preset]


def positive_int(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def non_negative_int(value: int, name: str) -> int:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def estimate_compression(
    ranks: Iterable[int],
    seq_len: int,
    kv_dim: int,
    sink_recency: int = 0,
    quant_bits: int = 4,
    original_bits: int = 16,
    residual_bits: int = 16,
) -> CompressionEstimate:
    rank_values = list(ranks)
    rank_count = len(rank_values)

    sink_recency = non_negative_int(sink_recency, "sink_recency")
    if sink_recency > seq_len:
        raise ValueError(
            f"sink_recency must be <= seq_len, got sink_recency={sink_recency}, "
            f"seq_len={seq_len}"
        )
    compressed_seq_len = seq_len - sink_recency

    original_per_matrix = seq_len * kv_dim * original_bits
    sink_recency_per_matrix = sink_recency * kv_dim * original_bits
    quantized_per_matrix = compressed_seq_len * kv_dim * quant_bits
    low_rank_total = sum(
        (compressed_seq_len + kv_dim + 1) * rank * residual_bits
        for rank in rank_values
    )

    return CompressionEstimate(
        rank_count=rank_count,
        seq_len=seq_len,
        sink_recency=sink_recency,
        compressed_seq_len=compressed_seq_len,
        kv_dim=kv_dim,
        quant_bits=quant_bits,
        residual_bits=residual_bits,
        original_bits=original_bits,
        original_total_bits=rank_count * original_per_matrix,
        compressed_total_bits=(
            rank_count * (sink_recency_per_matrix + quantized_per_matrix)
            + low_rank_total
        ),
    )


def write_per_row_output(
    output_path: Path,
    ranks: Iterable[int],
    seq_len: int,
    kv_dim: int,
    sink_recency: int,
    quant_bits: int,
    original_bits: int,
    residual_bits: int,
) -> None:
    if sink_recency > seq_len:
        raise ValueError(
            f"sink_recency must be <= seq_len, got sink_recency={sink_recency}, "
            f"seq_len={seq_len}"
        )
    compressed_seq_len = seq_len - sink_recency
    original_bits_per_matrix = seq_len * kv_dim * original_bits
    sink_recency_bits_per_matrix = sink_recency * kv_dim * original_bits
    quantized_bits_per_matrix = compressed_seq_len * kv_dim * quant_bits

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank_index",
                "adaptive_rank",
                "sink_recency",
                "compressed_seq_len",
                "original_bits",
                "compressed_bits",
                "compressed_percent_of_original",
                "reduction_percent",
            ]
        )
        for idx, rank in enumerate(ranks):
            compressed_bits = (
                sink_recency_bits_per_matrix
                + quantized_bits_per_matrix
                + (compressed_seq_len + kv_dim + 1) * rank * residual_bits
            )
            compressed_percent = 100.0 * compressed_bits / original_bits_per_matrix
            writer.writerow(
                [
                    idx,
                    rank,
                    sink_recency,
                    compressed_seq_len,
                    original_bits_per_matrix,
                    compressed_bits,
                    f"{compressed_percent:.6f}",
                    f"{100.0 - compressed_percent:.6f}",
                ]
            )


def format_bits(bits: float) -> str:
    bytes_value = bits / 8.0
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    unit_index = 0
    while bytes_value >= 1024 and unit_index < len(units) - 1:
        bytes_value /= 1024
        unit_index += 1
    return f"{bytes_value:.3f} {units[unit_index]}"


def main() -> None:
    args = parse_args()
    print(f"Using CSV path: {args.csv}")
    ranks = read_adaptive_ranks(args.csv)

    seq_len_estimated = args.seq_len is None
    seq_len = max(ranks) if seq_len_estimated else positive_int(args.seq_len, "--seq-len")
    kv_dim = resolve_kv_dim(args)

    estimate = estimate_compression(
        ranks=ranks,
        seq_len=seq_len,
        kv_dim=kv_dim,
        sink_recency=non_negative_int(args.sink_recency, "--sink-recency"),
        quant_bits=positive_int(args.quant_bits, "--quant-bits"),
        original_bits=positive_int(args.original_bits, "--original-bits"),
        residual_bits=positive_int(args.residual_bits, "--residual-bits"),
    )

    print(f"CSV: {args.csv}")
    print(f"Recorded K/V matrices: {estimate.rank_count}")
    if estimate.rank_count % 2 == 0:
        print(f"Approximate K/V pairs: {estimate.rank_count // 2}")
    print(
        "Adaptive rank stats: "
        f"min={min(ranks)}, max={max(ranks)}, "
        f"mean={mean(ranks):.3f}, median={median(ranks):.3f}"
    )
    print(f"Sequence length used: {seq_len}" + (" (estimated from max rank)" if seq_len_estimated else ""))
    print(f"Sink/recency tokens kept uncompressed: {estimate.sink_recency}")
    print(f"Compressed sequence length used: {estimate.compressed_seq_len}")
    print(f"Effective K/V dimension used: {kv_dim}")
    print(f"Original uncompressed size: {format_bits(estimate.original_total_bits)} (100.000%)")
    print(
        "Compressed size: "
        f"{format_bits(estimate.compressed_total_bits)} "
        f"({estimate.compressed_percent_of_original:.3f}% of original)"
    )
    print(f"Memory reduction: {estimate.reduction_percent:.3f}%")
    print(f"Memory Used: {100.0 - estimate.reduction_percent:.3f}%")
    print(f"Compression ratio: {estimate.compression_ratio:.3f}x")
    print(
        "Formula: compressed = FP16 sink/recency tokens + 4-bit quantized "
        "K/V base over (seq_len - sink_recency) + FP16 low-rank residual "
        "SVD factors ((seq_len - sink_recency) + kv_dim + 1) * adaptive_rank."
    )
    if seq_len_estimated:
        print(
            "Note: pass --seq-len for an exact estimate; max(adaptive_rank) is only "
            "a lower-bound shape estimate."
        )

    if args.per_row_output is not None:
        write_per_row_output(
            output_path=args.per_row_output,
            ranks=ranks,
            seq_len=seq_len,
            kv_dim=kv_dim,
            sink_recency=args.sink_recency,
            quant_bits=args.quant_bits,
            original_bits=args.original_bits,
            residual_bits=args.residual_bits,
        )
        print(f"Per-row details written to: {args.per_row_output}")


if __name__ == "__main__":
    main()
