import pytest

from variantshift.mutations import (
    Mutation,
    apply_variant,
    mutated_positions,
    parse_mutation,
    parse_variant,
    validate_against_sequence,
)


def test_parse_single_substitution() -> None:
    assert parse_mutation("A169C") == Mutation("A", 169, "C")


def test_parse_multi_mutant() -> None:
    parsed = parse_variant("A2C/D4E")
    assert [str(mutation) for mutation in parsed] == ["A2C", "D4E"]
    assert mutated_positions("A2C/D4E") == {2, 4}


def test_empty_code_is_wild_type() -> None:
    assert parse_variant("") == ()


@pytest.mark.parametrize("code", ["", "B12A", "A0C", "A12A", "A-1C", "p.A1C"])
def test_rejects_invalid_single_mutation(code: str) -> None:
    with pytest.raises(ValueError):
        parse_mutation(code)


def test_rejects_duplicate_position() -> None:
    with pytest.raises(ValueError, match="more than once"):
        parse_variant("A2C/A2D")


def test_validates_reference_sequence() -> None:
    validate_against_sequence("A1C/D3E", "ACD*")
    with pytest.raises(ValueError, match="Reference mismatch"):
        validate_against_sequence("C1A", "ACD*")


def test_applies_variant_and_preserves_terminal_stop() -> None:
    assert apply_variant("A1C/D3E", "ACD*") == "CCE*"

