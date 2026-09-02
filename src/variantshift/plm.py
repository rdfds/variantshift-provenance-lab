"""Optional protein-language-model scoring with explicit pretraining caveats."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .mutations import parse_variant, validate_against_sequence


def long_sequence_windows(
    length: int,
    *,
    window_size: int = 1022,
    overlap: int = 256,
) -> list[tuple[int, int]]:
    """Return deterministic zero-based half-open windows covering a protein sequence."""
    if length < 1:
        raise ValueError("Protein sequence cannot be empty")
    if not 0 <= overlap < window_size:
        raise ValueError("Overlap must lie in [0, window_size)")
    if length <= window_size:
        return [(0, length)]
    step = window_size - overlap
    starts = list(range(0, length - window_size + 1, step))
    final_start = length - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(start, start + window_size) for start in starts]


def assign_positions_to_windows(
    length: int,
    positions: Sequence[int],
    *,
    window_size: int = 1022,
    overlap: int = 256,
) -> dict[tuple[int, int], list[int]]:
    """Assign each one-based position to the window with the most boundary context."""
    windows = long_sequence_windows(length, window_size=window_size, overlap=overlap)
    assignments: dict[tuple[int, int], list[int]] = {window: [] for window in windows}
    for position in sorted({int(value) for value in positions}):
        if not 1 <= position <= length:
            raise ValueError(f"Residue position {position} exceeds sequence length {length}")
        zero_based = position - 1
        containing = [window for window in windows if window[0] <= zero_based < window[1]]
        if not containing:
            raise RuntimeError(f"No long-sequence window contains residue position {position}")
        # ``max`` retains the earlier window on a tie because the windows are ordered and the
        # negative start coordinate is the second tuple element.
        chosen = max(
            containing,
            key=lambda window: (
                min(zero_based - window[0], window[1] - 1 - zero_based),
                -window[0],
            ),
        )
        assignments[chosen].append(position)
    return {window: values for window, values in assignments.items() if values}


def esm2_position_log_probabilities(
    sequence: str,
    positions: Sequence[int],
    *,
    model_name: str = "esm2_t6_8M_UR50D",
    strategy: str = "masked-marginal",
    device: str | None = None,
    batch_size: int = 16,
    window_size: int = 1022,
    overlap: int = 256,
) -> tuple[dict[int, np.ndarray], Mapping[str, int]]:
    """Compute amino-acid log probabilities at selected positions of any-length sequences.

    Masked-marginal scoring masks one residue at a time, batched within the deterministic window
    assigned by :func:`assign_positions_to_windows`.  Wild-type marginals use the same windows so
    the secondary comparison changes only the masking strategy.
    """
    try:
        import esm
        import torch
    except ImportError as error:
        raise RuntimeError("ESM scoring requires the optional 'plm' dependencies") from error
    if strategy not in {"masked-marginal", "wild-type-marginal"}:
        raise ValueError("ESM-2 strategy must be masked-marginal or wild-type-marginal")
    if batch_size < 1:
        raise ValueError("ESM-2 batch size must be positive")
    sequence = sequence.removesuffix("*").upper()
    if set(sequence).difference(set("ACDEFGHIKLMNPQRSTVWY")):
        raise ValueError("ESM-2 external scoring requires the 20 standard amino acids")
    loader = getattr(esm.pretrained, model_name, None)
    if loader is None:
        raise ValueError(f"Unsupported fair-esm model: {model_name}")
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    model, alphabet = loader()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    assignments = assign_positions_to_windows(
        len(sequence),
        positions,
        window_size=window_size,
        overlap=overlap,
    )
    results: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for (start, end), assigned_positions in assignments.items():
            chunk = sequence[start:end]
            _, _, base_tokens = batch_converter([("external", chunk)])
            if strategy == "wild-type-marginal":
                logits = model(base_tokens.to(device), repr_layers=[], return_contacts=False)[
                    "logits"
                ]
                log_probabilities = torch.log_softmax(logits[0], dim=-1)
                for position in assigned_positions:
                    local_token = position - start
                    results[position] = log_probabilities[local_token].float().cpu().numpy()
                continue
            for offset in range(0, len(assigned_positions), batch_size):
                current = assigned_positions[offset : offset + batch_size]
                tokens = base_tokens.repeat(len(current), 1)
                local_tokens = torch.tensor(
                    [position - start for position in current], dtype=torch.long
                )
                tokens[torch.arange(len(current)), local_tokens] = alphabet.mask_idx
                logits = model(tokens.to(device), repr_layers=[], return_contacts=False)["logits"]
                rows = torch.arange(len(current), device=device)
                selected = logits[rows, local_tokens.to(device)]
                log_probabilities = torch.log_softmax(selected, dim=-1).float().cpu().numpy()
                results.update(
                    {
                        position: log_probabilities[index]
                        for index, position in enumerate(current)
                    }
                )
    if set(results) != {int(position) for position in positions}:
        raise RuntimeError("ESM-2 scoring did not return every requested residue position")
    token_index = {
        amino_acid: alphabet.get_idx(amino_acid) for amino_acid in "ACDEFGHIKLMNPQRSTVWY"
    }
    return results, token_index


def score_single_substitutions(
    position_log_probabilities: Mapping[int, np.ndarray],
    variants: Iterable[str],
    token_index: Mapping[str, int],
) -> pd.DataFrame:
    """Score canonical single substitutions from per-position ESM log probabilities."""
    codes = list(dict.fromkeys(str(code) for code in variants))
    scores = []
    for code in codes:
        mutations = parse_variant(code)
        if len(mutations) != 1 or mutations[0].is_stop:
            raise ValueError(f"Expected one missense substitution, received {code}")
        mutation = mutations[0]
        probabilities = position_log_probabilities.get(mutation.position)
        if probabilities is None:
            raise ValueError(f"Missing ESM probabilities for position {mutation.position}")
        scores.append(
            float(
                probabilities[token_index[mutation.alternate]]
                - probabilities[token_index[mutation.reference]]
            )
        )
    return pd.DataFrame({"mutation_codes": codes, "prediction": scores})


def score_variant_log_odds(
    log_probabilities: np.ndarray,
    codes: str,
    token_index: Mapping[str, int],
) -> float:
    """Sum alternate-vs-reference log odds under wild-type contextual logits.

    ``log_probabilities`` is indexed by zero-based protein position and vocabulary token.
    The operation is additive for multi-mutants and does not model epistasis.
    """
    score = 0.0
    for mutation in parse_variant(codes):
        if mutation.is_stop:
            raise ValueError("ESM scoring does not support nonsense mutations")
        position = mutation.position - 1
        score += float(
            log_probabilities[position, token_index[mutation.alternate]]
            - log_probabilities[position, token_index[mutation.reference]]
        )
    return score


def esm2_wild_type_marginals(
    wild_type_sequence: str,
    variants: Iterable[str],
) -> pd.DataFrame:
    """Score variants with the compact ESM-2 8M model in a single forward pass.

    This is the fast wild-type-marginal strategy, not masked-marginal pseudo-likelihood.
    Install the optional dependency group with ``pip install -e '.[plm]'``.
    """
    try:
        import esm
        import torch
    except ImportError as error:
        raise RuntimeError("ESM scoring requires the optional 'plm' dependencies") from error

    sequence = wild_type_sequence.removesuffix("*")
    codes = list(dict.fromkeys(map(str, variants)))
    for code in codes:
        validate_against_sequence(code, sequence)

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("TEV", sequence)])
    with torch.no_grad():
        logits = model(tokens, repr_layers=[], return_contacts=False)["logits"]
        # Remove BOS and EOS positions so row zero corresponds to residue one.
        log_probabilities = torch.log_softmax(logits[0, 1 : len(sequence) + 1], dim=-1)
    probabilities = log_probabilities.cpu().numpy()
    token_index = {amino_acid: alphabet.get_idx(amino_acid) for amino_acid in sequence}
    token_index.update(
        {
            mutation.alternate: alphabet.get_idx(mutation.alternate)
            for code in codes
            for mutation in parse_variant(code)
            if not mutation.is_stop
        }
    )
    return pd.DataFrame(
        {
            "mutation_codes": codes,
            "esm2_8m_wt_marginal": [
                score_variant_log_odds(probabilities, code, token_index) for code in codes
            ],
        }
    )
