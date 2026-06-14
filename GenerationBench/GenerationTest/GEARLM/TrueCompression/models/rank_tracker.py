import csv
import os
from typing import Iterable, List, Union
import torch

# Shared adaptive rank list across TrueCompression imports
_adaptive_rank_distribution: List[int] = []


def get_rank_distribution() -> List[int]:
    """Return the recorded adaptive ranks."""
    return _adaptive_rank_distribution


def clear_rank_distribution() -> None:
    """Reset the recorded adaptive ranks."""
    global _adaptive_rank_distribution
    _adaptive_rank_distribution = []


def append_rank_distribution(ranks: Union[int, Iterable[int], torch.Tensor]) -> None:
    """Append one or more adaptive rank values to the shared tracker."""
    global _adaptive_rank_distribution
    if isinstance(ranks, torch.Tensor):
        ranks = ranks.detach().cpu().tolist()
    if isinstance(ranks, (list, tuple)):
        _adaptive_rank_distribution.extend(int(rank) for rank in ranks)
    else:
        _adaptive_rank_distribution.append(int(ranks))


def save_rank_distribution_to_csv(filename: str = "adaptive_rank_distribution.csv") -> None:
    """Write the recorded adaptive ranks to a CSV file."""
    if not _adaptive_rank_distribution:
        print("Warning: No adaptive rank data to save.")
        return

    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    try:
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["rank_index", "adaptive_rank"])
            for idx, rank in enumerate(_adaptive_rank_distribution):
                writer.writerow([idx, rank])
    except Exception as exc:
        print(f"Error saving rank distribution to CSV: {exc}")
