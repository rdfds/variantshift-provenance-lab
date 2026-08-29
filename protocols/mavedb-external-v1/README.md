# Locked-box external validation protocol

This directory is the methods-before-outcomes boundary for VariantShift's external validation.
It was generated without requesting any MaveDB score table.

The panel contains every public MaveDB score set that, at the freeze timestamp:

1. was published after ProteinGym v1.3 (April 28, 2025) and by August 29, 2026;
2. declared at least 500 variants and exactly one protein target of 40–2,000 residues;
3. used only standard amino acids after normalizing an optional terminal stop; and
4. had no ProteinGym target-family hit at ≥30% sequence identity and ≥80% bidirectional
   coverage in the exhaustive MMseqs2 audit.

All 65 metadata candidates remain in `metadata-registry.csv`. Twenty assays fail the family rule;
the remaining 45 assays across 17 named targets and 18 distinct target sequences form the frozen
panel. `detailed-metadata-snapshot.json` preserves the public calibration metadata used to orient
scores before outcomes. `protocol.json` fixes the models, scoring strategies, inclusion rules,
estimands, bootstrap unit, and success criterion.

The next commit must contain this directory with `outcomes_accessed: false`. Only after that commit
is pushed may `variantshift mavedb-download-external` request the selected score tables. Every
subsequent inclusion and exclusion must be published.

