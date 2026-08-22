"""Parsing and validation for compact amino-acid mutation strings."""

from __future__ import annotations

import re
from dataclasses import dataclass

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
SUBSTITUTION_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z*])$")


@dataclass(frozen=True, order=True)
class Mutation:
    reference: str
    position: int
    alternate: str

    def __post_init__(self) -> None:
        if self.reference not in AMINO_ACIDS:
            raise ValueError(f"Invalid reference amino acid: {self.reference}")
        if self.alternate not in AMINO_ACIDS | {"*"}:
            raise ValueError(f"Invalid alternate amino acid: {self.alternate}")
        if self.position < 1:
            raise ValueError("Mutation positions are one-indexed and must be positive")
        if self.reference == self.alternate:
            raise ValueError("A substitution must change the amino acid")

    @property
    def is_stop(self) -> bool:
        return self.alternate == "*"

    def __str__(self) -> str:
        return f"{self.reference}{self.position}{self.alternate}"


def parse_mutation(code: str) -> Mutation:
    """Parse a single substitution such as ``A169C``."""
    match = SUBSTITUTION_PATTERN.fullmatch(code.strip())
    if not match:
        raise ValueError(f"Unsupported mutation code: {code!r}")
    reference, position, alternate = match.groups()
    return Mutation(reference, int(position), alternate)


def parse_variant(codes: str) -> tuple[Mutation, ...]:
    """Parse slash-delimited substitutions; an empty string denotes wild type."""
    codes = codes.strip()
    if not codes:
        return ()
    mutations = tuple(parse_mutation(code) for code in codes.split("/"))
    positions = [mutation.position for mutation in mutations]
    if len(positions) != len(set(positions)):
        raise ValueError(f"Variant mutates a residue more than once: {codes!r}")
    return mutations


def mutated_positions(codes: str) -> frozenset[int]:
    return frozenset(mutation.position for mutation in parse_variant(codes))


def validate_against_sequence(codes: str, wild_type_sequence: str) -> None:
    """Raise when encoded reference residues disagree with the wild-type sequence."""
    sequence = wild_type_sequence.removesuffix("*")
    for mutation in parse_variant(codes):
        if mutation.position > len(sequence):
            raise ValueError(
                f"Position {mutation.position} exceeds sequence length {len(sequence)}"
            )
        observed = sequence[mutation.position - 1]
        if observed != mutation.reference:
            raise ValueError(
                f"Reference mismatch at {mutation.position}: "
                f"mutation says {mutation.reference}, sequence has {observed}"
            )


def apply_variant(codes: str, wild_type_sequence: str) -> str:
    """Apply substitutions to a protein sequence."""
    had_terminal_stop = wild_type_sequence.endswith("*")
    residues = list(wild_type_sequence.removesuffix("*"))
    validate_against_sequence(codes, wild_type_sequence)
    for mutation in parse_variant(codes):
        residues[mutation.position - 1] = mutation.alternate
    result = "".join(residues)
    return result + ("*" if had_terminal_stop and "*" not in result else "")

