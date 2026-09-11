import pytest

from variantshift.structure_inputs import crop_alphafold_pdb, crop_exact_pdb_chain


def _atom(serial: int, atom: str, residue: str, position: int, plddt: float) -> str:
    return (
        f"ATOM  {serial:5d} {atom:^4s} {residue:>3s} A{position:4d}    "
        f"{float(position):8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{plddt:6.2f}           C"
    )


def test_crop_alphafold_pdb_validates_sequence_and_renumbers() -> None:
    lines = [
        _atom(1, "CA", "ALA", 9, 50.0),
        _atom(2, "N", "CYS", 10, 70.0),
        _atom(3, "CA", "CYS", 10, 80.0),
        _atom(4, "CA", "ASP", 11, 90.0),
        _atom(5, "CA", "GLU", 12, 60.0),
    ]
    cropped, mean_plddt = crop_alphafold_pdb(
        ("\n".join(lines) + "\n").encode(), start=10, sequence="CD"
    )
    text = cropped.decode()
    assert " A   1" in text
    assert " A   2" in text
    assert "ALA" not in text
    assert "GLU" not in text
    assert mean_plddt == pytest.approx(85.0)


def test_crop_alphafold_pdb_rejects_sequence_drift() -> None:
    payload = (_atom(1, "CA", "CYS", 10, 80.0) + "\n").encode()
    with pytest.raises(ValueError, match="differs"):
        crop_alphafold_pdb(payload, start=10, sequence="D")


def _pdb_atom(serial: int, atom: str, residue: str, chain: str, position: int) -> str:
    return (
        f"ATOM  {serial:5d} {atom:^4s} {residue:>3s} {chain}{position:4d}    "
        f"{float(position):8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{80.0:6.2f}           C"
    )


def test_crop_exact_pdb_chain_selects_and_renumbers_exact_backbone() -> None:
    lines = []
    serial = 0
    for chain in ("B", "A"):
        for position, residue in ((10, "ALA"), (12, "CYS")):
            for atom in ("N", "CA", "C"):
                serial += 1
                lines.append(_pdb_atom(serial, atom, residue, chain, position))
    cropped, source_chain = crop_exact_pdb_chain(
        ("\n".join(lines) + "\n").encode(), sequence="AC"
    )
    text = cropped.decode()
    assert source_chain == "A"
    assert "EXACT-CHAIN A" in text
    assert " A   1 " in text
    assert " A   2 " in text
    assert " B  10 " not in text


def test_crop_exact_pdb_chain_rejects_missing_backbone() -> None:
    payload = (_pdb_atom(1, "CA", "ALA", "A", 1) + "\n").encode()
    with pytest.raises(ValueError, match="complete N/CA/C backbone"):
        crop_exact_pdb_chain(payload, sequence="A")
