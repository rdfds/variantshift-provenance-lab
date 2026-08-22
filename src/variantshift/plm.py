"""Optional protein-language-model scoring with explicit pretraining caveats."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .mutations import parse_variant, validate_against_sequence


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

