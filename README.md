# VariantShift

**When does a protein mutation predictor actually generalize?**

VariantShift is a leakage-aware benchmark for protein variant-effect models. It compares
ordinary random splits with biologically harder tests: unseen residue positions, increased
mutational depth, and transfer between experimental conditions.

The initial case study uses the Align Foundation's TEV protease GROQ-seq release: 18,486
variants measured across 24 conditions at NIST's Living Measurement Systems Foundry.

## Research questions

1. How much does random splitting overstate performance?
2. Which models retain rank accuracy at residue positions absent from training?
3. Do single-mutant models transfer to combinatorial variants?
4. Is model confidence calibrated when the biological distribution shifts?

## Status

The repository is under active development. The first release will include deterministic
data validation, biochemical baselines, leakage-aware splitters, uncertainty estimates,
and a standalone benchmark report.

## Data

VariantShift never vendors the source measurements. Downloading the dataset requires
explicitly accepting the provider's data-use agreement.

Dataset: [TEV Protease — Pilot SSVL and epPCR Libraries](https://data.alignbio.org/groqseq/groqseq-014/)

## License

Code is released under the MIT License. The TEV dataset has separate terms from its provider.

