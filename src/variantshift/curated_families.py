"""Curated Pfam/InterPro validation for ProteinGym family holdouts."""

from __future__ import annotations

import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .family_clusters import _DisjointSet
from .proteingym import read_reference_index

INTERPRO_API = "https://www.ebi.ac.uk/interpro/api"
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"
USER_AGENT = "VariantShift/0.4 curated-family research audit"


@dataclass(frozen=True)
class CuratedFamilyResult:
    assignments: pd.DataFrame
    clan_assignments: pd.DataFrame
    uniprot_mapping: pd.DataFrame
    coordinate_mapping: pd.DataFrame
    domain_overlaps: pd.DataFrame
    curated_edges: pd.DataFrame
    clan_edges: pd.DataFrame
    audit: pd.DataFrame


def _request_bytes(url: str, *, attempts: int = 5) -> tuple[bytes, dict[str, str]]:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(
                url,
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            )
            with urlopen(request, timeout=90) as response:
                return response.read(), {
                    key.lower(): value for key, value in response.headers.items()
                }
        except (HTTPError, URLError, TimeoutError) as current:
            error = current
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"External annotation request failed after {attempts} attempts: {url}"
    ) from error


def _cached_bytes(url: str, cache_path: Path) -> tuple[bytes, dict[str, str]]:
    cache_path = Path(cache_path)
    headers_path = cache_path.with_suffix(cache_path.suffix + ".headers.json")
    if cache_path.is_file():
        headers = json.loads(headers_path.read_text()) if headers_path.is_file() else {}
        return cache_path.read_bytes(), headers
    payload, headers = _request_bytes(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(payload)
    headers_path.write_text(json.dumps(headers, indent=2, sort_keys=True) + "\n")
    return payload, headers


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _eligible_reference(reference_path: Path, eligibility: pd.DataFrame) -> pd.DataFrame:
    reference = read_reference_index(reference_path)
    eligible_ids = set(
        eligibility.loc[eligibility["eligible"].astype(bool), "assay_id"].astype(str)
    )
    selected = reference.loc[
        reference["DMS_id"].astype(str).isin(eligible_ids),
        ["DMS_id", "UniProt_ID", "target_seq", "MSA_start", "MSA_end"],
    ].copy()
    if len(selected) != len(eligible_ids):
        raise ValueError("Eligible assays are missing from the ProteinGym reference")
    return (
        selected.rename(columns={"DMS_id": "assay_id", "UniProt_ID": "uniprot_id"})
        .sort_values("assay_id")
        .reset_index(drop=True)
    )


def _resolve_uniprot(entry_names: list[str], cache_dir: Path) -> tuple[pd.DataFrame, str]:
    frames: list[pd.DataFrame] = []
    for offset in range(0, len(entry_names), 20):
        chunk = entry_names[offset : offset + 20]
        query = " OR ".join(f"id:{entry}" for entry in chunk)
        parameters = urlencode(
            {
                "query": f"({query})",
                "format": "tsv",
                "fields": "accession,id,sequence",
                "size": "500",
            }
        )
        key = _digest("|".join(chunk))[:16]
        payload, _ = _cached_bytes(
            f"{UNIPROT_API}?{parameters}", cache_dir / "uniprot" / f"{key}.tsv"
        )
        frames.append(pd.read_csv(io.BytesIO(payload), sep="\t"))
    mapped = pd.concat(frames, ignore_index=True).drop_duplicates("Entry Name")
    mapped = mapped.rename(
        columns={
            "Entry": "uniprot_accession",
            "Entry Name": "uniprot_id",
            "Sequence": "canonical_sequence",
        }
    )
    mapped = mapped.loc[mapped["canonical_sequence"].notna()].copy()
    mapped["canonical_length"] = mapped["canonical_sequence"].str.len()
    mapped["canonical_sequence_sha256"] = mapped["canonical_sequence"].map(_digest)
    api_payload, _ = _cached_bytes(f"{INTERPRO_API}/", cache_dir / "interpro" / "api-metadata.json")
    api_metadata = json.loads(api_payload)
    uniprot_version = str(api_metadata["databases"]["uniprot"]["version"])
    return mapped.sort_values("uniprot_id").reset_index(drop=True), uniprot_version


def map_assayed_region(
    target_sequence: str,
    canonical_sequence: str,
    assay_start: int,
    assay_end: int,
) -> dict[str, int | float | None]:
    """Map an assayed interval to one contiguous UniProt segment using exact anchors."""
    if not 1 <= assay_start <= assay_end <= len(target_sequence):
        raise ValueError("Assayed coordinates exceed the ProteinGym target sequence")
    matcher = SequenceMatcher(None, target_sequence, canonical_sequence, autojunk=False)
    query_start = assay_start - 1
    query_end = assay_end
    candidate_offsets = {block.b - block.a for block in matcher.get_matching_blocks() if block.size}
    if not candidate_offsets:
        candidate_offsets = {0}

    def matched_positions(offset: int) -> list[int]:
        return [
            query_index + offset + 1
            for query_index in range(query_start, query_end)
            if 0 <= query_index + offset < len(canonical_sequence)
            and target_sequence[query_index] == canonical_sequence[query_index + offset]
        ]

    _offset, canonical_positions = max(
        ((offset, matched_positions(offset)) for offset in candidate_offsets),
        key=lambda item: (len(item[1]), -abs(item[0])),
    )
    assay_length = query_end - query_start
    return {
        "assayed_residues": assay_length,
        "mapped_residues": len(canonical_positions),
        "mapping_coverage": len(canonical_positions) / assay_length,
        "canonical_start": min(canonical_positions) if canonical_positions else None,
        "canonical_end": max(canonical_positions) if canonical_positions else None,
    }


def _coordinate_mappings(reference: pd.DataFrame, uniprot: pd.DataFrame) -> pd.DataFrame:
    merged = reference.merge(
        uniprot[["uniprot_id", "uniprot_accession", "canonical_sequence"]],
        on="uniprot_id",
        how="left",
        validate="many_to_one",
    )
    rows = []
    for row in merged.itertuples(index=False):
        if not isinstance(row.canonical_sequence, str):
            mapping = {
                "assayed_residues": int(row.MSA_end) - int(row.MSA_start) + 1,
                "mapped_residues": 0,
                "mapping_coverage": 0.0,
                "canonical_start": None,
                "canonical_end": None,
            }
        else:
            mapping = map_assayed_region(
                str(row.target_seq),
                row.canonical_sequence,
                int(row.MSA_start),
                int(row.MSA_end),
            )
        rows.append(
            {
                "assay_id": row.assay_id,
                "uniprot_id": row.uniprot_id,
                "uniprot_accession": row.uniprot_accession,
                "target_sequence_sha256": _digest(str(row.target_seq)),
                "target_length": len(str(row.target_seq)),
                "assay_start": int(row.MSA_start),
                "assay_end": int(row.MSA_end),
                **mapping,
            }
        )
    return pd.DataFrame(rows).sort_values("assay_id").reset_index(drop=True)


def _protein_pfams(accession: str, uniprot_id: str, cache_dir: Path) -> list[dict[str, object]]:
    payload, _ = _cached_bytes(
        f"{INTERPRO_API}/entry/pfam/protein/uniprot/{accession}/?page_size=200",
        cache_dir / "interpro" / "proteins" / f"{accession}.json",
    )
    # InterPro returns HTTP 204 with an empty body for proteins without a Pfam
    # assignment.  That is an audited negative result, not malformed JSON.
    if not payload.strip():
        return []
    response = json.loads(payload)
    if response.get("next"):
        raise RuntimeError(f"Pfam response requires unexpected pagination: {accession}")
    rows: list[dict[str, object]] = []
    for result in response.get("results", []):
        metadata = result["metadata"]
        for protein in result.get("proteins", []):
            for location in protein.get("entry_protein_locations", []):
                for fragment in location.get("fragments", []):
                    rows.append(
                        {
                            "uniprot_id": uniprot_id,
                            "uniprot_accession": accession,
                            "pfam_accession": metadata["accession"],
                            "pfam_name": metadata["name"],
                            "interpro_accession": metadata.get("integrated"),
                            "domain_start": int(fragment["start"]),
                            "domain_end": int(fragment["end"]),
                            "domain_score": location.get("score"),
                        }
                    )
    return rows


def _pfam_clan(pfam_accession: str, cache_dir: Path) -> dict[str, object]:
    payload, _ = _cached_bytes(
        f"{INTERPRO_API}/entry/pfam/{pfam_accession}/",
        cache_dir / "interpro" / "pfam" / f"{pfam_accession}.json",
    )
    metadata = json.loads(payload)["metadata"]
    clan = metadata.get("set_info") or {}
    return {
        "pfam_accession": pfam_accession,
        "clan_accession": clan.get("accession"),
        "clan_name": clan.get("name"),
    }


def _fetch_domains(
    uniprot: pd.DataFrame, cache_dir: Path, workers: int
) -> tuple[pd.DataFrame, str, str]:
    proteins = [
        (str(row.uniprot_accession), str(row.uniprot_id), cache_dir)
        for row in uniprot.itertuples(index=False)
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        nested = list(executor.map(lambda args: _protein_pfams(*args), proteins))
    domain_rows = [row for group in nested for row in group]
    domains = pd.DataFrame(domain_rows)
    if domains.empty:
        raise ValueError("No curated Pfam annotations were returned")
    pfams = sorted(domains["pfam_accession"].astype(str).unique())
    with ThreadPoolExecutor(max_workers=workers) as executor:
        clan_rows = list(executor.map(lambda pfam: _pfam_clan(pfam, cache_dir), pfams))
    domains = domains.merge(
        pd.DataFrame(clan_rows), on="pfam_accession", how="left", validate="many_to_one"
    )
    metadata_payload, metadata_headers = _cached_bytes(
        f"{INTERPRO_API}/", cache_dir / "interpro" / "api-metadata.json"
    )
    metadata = json.loads(metadata_payload)
    version_path = cache_dir / "interpro" / "api-version.json"
    if "interpro-version" in metadata_headers:
        interpro_version = str(metadata_headers["interpro-version"])
        version_path.write_text(
            json.dumps({"interpro_version": interpro_version}, sort_keys=True) + "\n"
        )
    elif version_path.is_file():
        interpro_version = str(json.loads(version_path.read_text())["interpro_version"])
    else:
        interpro_version = "unknown"
    pfam_version = str(metadata["databases"]["pfam"]["version"])
    return domains, interpro_version, pfam_version


def _domain_overlaps(
    coordinates: pd.DataFrame,
    domains: pd.DataFrame,
    *,
    minimum_mapping_coverage: float,
    minimum_overlap: float,
) -> pd.DataFrame:
    rows = []
    for coordinate in coordinates.itertuples(index=False):
        candidates = domains.loc[domains["uniprot_id"].eq(coordinate.uniprot_id)]
        for domain in candidates.itertuples(index=False):
            if coordinate.mapping_coverage < minimum_mapping_coverage:
                overlap = 0
                fraction = 0.0
            else:
                overlap = max(
                    0,
                    min(int(coordinate.canonical_end), domain.domain_end)
                    - max(int(coordinate.canonical_start), domain.domain_start)
                    + 1,
                )
                assay_span = int(coordinate.canonical_end) - int(coordinate.canonical_start) + 1
                domain_span = domain.domain_end - domain.domain_start + 1
                fraction = overlap / min(assay_span, domain_span)
            rows.append(
                {
                    "assay_id": coordinate.assay_id,
                    "uniprot_id": coordinate.uniprot_id,
                    "uniprot_accession": coordinate.uniprot_accession,
                    "mapping_coverage": coordinate.mapping_coverage,
                    "canonical_start": coordinate.canonical_start,
                    "canonical_end": coordinate.canonical_end,
                    "pfam_accession": domain.pfam_accession,
                    "pfam_name": domain.pfam_name,
                    "clan_accession": domain.clan_accession,
                    "clan_name": domain.clan_name,
                    "domain_start": domain.domain_start,
                    "domain_end": domain.domain_end,
                    "domain_score": domain.domain_score,
                    "overlap_residues": overlap,
                    "overlap_fraction_of_shorter": fraction,
                    "qualifies_curated_domain": (
                        coordinate.mapping_coverage >= minimum_mapping_coverage
                        and fraction >= minimum_overlap
                    ),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["assay_id", "pfam_accession", "domain_start"])
        .reset_index(drop=True)
    )


def cluster_from_curated_domains(
    base_assignments: pd.DataFrame,
    domain_overlaps: pd.DataFrame,
    *,
    grouping: str = "pfam_family",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Union existing families through shared qualifying Pfam families or clans."""
    if grouping not in {"pfam_family", "pfam_clan"}:
        raise ValueError("Curated grouping must be 'pfam_family' or 'pfam_clan'")
    required = {"uniprot_id", "family_id"}
    missing = required.difference(base_assignments.columns)
    if missing:
        raise ValueError(f"Base assignments are missing: {', '.join(sorted(missing))}")
    if base_assignments["uniprot_id"].duplicated().any():
        raise ValueError("Each UniProt ID must have one base family")
    proteins = sorted(base_assignments["uniprot_id"].astype(str))
    disjoint = _DisjointSet(proteins)
    for _, group in base_assignments.groupby("family_id"):
        members = sorted(group["uniprot_id"].astype(str))
        for member in members[1:]:
            disjoint.union(members[0], member)
    qualifying = domain_overlaps.loc[domain_overlaps["qualifies_curated_domain"]].copy()
    qualifying["curated_group_id"] = (
        qualifying["pfam_accession"]
        if grouping == "pfam_family"
        else qualifying["clan_accession"].fillna(qualifying["pfam_accession"])
    )
    edges = []
    for group_id, group in qualifying.groupby("curated_group_id"):
        members = sorted(group["uniprot_id"].astype(str).unique())
        for left_index, left in enumerate(members):
            for right in members[left_index + 1 :]:
                disjoint.union(left, right)
                edges.append({"curated_group_id": group_id, "protein_a": left, "protein_b": right})
    components: dict[str, list[str]] = {}
    for protein in proteins:
        components.setdefault(disjoint.find(protein), []).append(protein)
    family_by_protein: dict[str, str] = {}
    members_by_family: dict[str, tuple[str, ...]] = {}
    for members in components.values():
        ordered = tuple(sorted(members))
        digest = _digest("|".join(ordered))[:12]
        family_id = f"cssf_{digest}" if grouping == "pfam_family" else f"cssc_{digest}"
        members_by_family[family_id] = ordered
        family_by_protein.update({protein: family_id for protein in ordered})
    assignments = base_assignments.rename(columns={"family_id": "base_family_id"}).copy()
    assignments["family_id"] = assignments["uniprot_id"].map(family_by_protein)
    assignments["family_size"] = assignments["family_id"].map(
        {family: len(members) for family, members in members_by_family.items()}
    )
    assignments["family_members"] = assignments["family_id"].map(
        {family: ";".join(members) for family, members in members_by_family.items()}
    )
    preferred = [
        "uniprot_id",
        "family_id",
        "family_size",
        "family_members",
        "base_family_id",
    ]
    assignments = (
        assignments[
            preferred + [column for column in assignments.columns if column not in preferred]
        ]
        .sort_values(["family_id", "uniprot_id"])
        .reset_index(drop=True)
    )
    edge_frame = pd.DataFrame(edges, columns=["curated_group_id", "protein_a", "protein_b"])
    if not edge_frame.empty:
        edge_frame["protein_a_family_id"] = edge_frame["protein_a"].map(family_by_protein)
        edge_frame["protein_b_family_id"] = edge_frame["protein_b"].map(family_by_protein)
        if edge_frame["protein_a_family_id"].ne(edge_frame["protein_b_family_id"]).any():
            raise RuntimeError("A curated homology edge crosses final family clusters")
    return assignments, edge_frame


def build_curated_family_clusters(
    reference_path: Path,
    eligibility: pd.DataFrame,
    base_assignments: pd.DataFrame,
    cache_dir: Path,
    *,
    minimum_mapping_coverage: float = 0.80,
    minimum_overlap: float = 0.50,
    workers: int = 4,
) -> CuratedFamilyResult:
    """Add current Pfam family/clan evidence to sequence-structure families."""
    if not 0 < minimum_mapping_coverage <= 1 or not 0 < minimum_overlap <= 1:
        raise ValueError("Mapping and overlap thresholds must be in (0, 1]")
    if workers < 1:
        raise ValueError("Worker count must be positive")
    reference = _eligible_reference(reference_path, eligibility)
    proteins = sorted(reference["uniprot_id"].astype(str).unique())
    uniprot, uniprot_version = _resolve_uniprot(proteins, Path(cache_dir))
    coordinates = _coordinate_mappings(reference, uniprot)
    domains, interpro_version, pfam_version = _fetch_domains(uniprot, Path(cache_dir), workers)
    overlaps = _domain_overlaps(
        coordinates,
        domains,
        minimum_mapping_coverage=minimum_mapping_coverage,
        minimum_overlap=minimum_overlap,
    )
    assignments, edges = cluster_from_curated_domains(
        base_assignments, overlaps, grouping="pfam_family"
    )
    clan_assignments, clan_edges = cluster_from_curated_domains(
        base_assignments, overlaps, grouping="pfam_clan"
    )
    qualifying = overlaps.loc[overlaps["qualifies_curated_domain"]]
    shared_audit = {
        "interpro_version": interpro_version,
        "pfam_version": pfam_version,
        "uniprot_version": uniprot_version,
        "eligible_assays": reference["assay_id"].nunique(),
        "proteins": len(proteins),
        "mapped_uniprot_entries": uniprot["uniprot_id"].nunique(),
        "coordinate_mapped_assays": int(
            coordinates["mapping_coverage"].ge(minimum_mapping_coverage).sum()
        ),
        "coordinate_mapped_proteins": int(
            coordinates.loc[
                coordinates["mapping_coverage"].ge(minimum_mapping_coverage),
                "uniprot_id",
            ].nunique()
        ),
        "qualifying_domain_overlaps": len(qualifying),
        "curated_annotated_assays": qualifying["assay_id"].nunique(),
        "curated_annotated_proteins": qualifying["uniprot_id"].nunique(),
        "qualifying_pfam_families": qualifying["pfam_accession"].nunique(),
        "qualifying_pfam_clans": qualifying["clan_accession"].nunique(),
        "minimum_coordinate_mapping_coverage": minimum_mapping_coverage,
        "minimum_domain_overlap_fraction_of_shorter": minimum_overlap,
        "base_family_count": base_assignments["family_id"].nunique(),
    }
    audit_rows = []
    for grouping, grouped_assignments, grouped_edges in (
        ("pfam_family", assignments, edges),
        ("pfam_clan_sensitivity", clan_assignments, clan_edges),
    ):
        sizes = grouped_assignments.groupby("family_id")["uniprot_id"].nunique()
        audit_rows.append(
            {
                "method": grouping,
                **shared_audit,
                "curated_family_count": len(sizes),
                "families_merged_by_curated_annotations": (
                    base_assignments["family_id"].nunique() - len(sizes)
                ),
                "singleton_families": int(sizes.eq(1).sum()),
                "multi_protein_families": int(sizes.gt(1).sum()),
                "proteins_in_multi_protein_families": int(sizes.loc[sizes.gt(1)].sum()),
                "largest_family_size": int(sizes.max()),
                "curated_pair_edges": len(grouped_edges),
                "cross_family_curated_edges": int(
                    grouped_edges["protein_a_family_id"]
                    .ne(grouped_edges["protein_b_family_id"])
                    .sum()
                    if not grouped_edges.empty
                    else 0
                ),
            }
        )
    audit = pd.DataFrame(audit_rows)
    public_mapping = uniprot.drop(columns=["canonical_sequence"])
    return CuratedFamilyResult(
        assignments,
        clan_assignments,
        public_mapping,
        coordinates,
        overlaps,
        edges,
        clan_edges,
        audit,
    )
