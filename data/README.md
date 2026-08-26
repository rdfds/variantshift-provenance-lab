# Data directory

Raw measurements are intentionally excluded from version control.

```bash
variantshift download data/raw --accept-data-use-agreement
```

Before downloading, review the [Align Foundation data-use agreement](https://data.alignbio.org/agreement/data-use-agreement-v1.html).
The expected file is `data/raw/TEV_Pilot_SSVL_EP_output_v1.1.csv`.

The independent validation uses the public ProteinGym v1.3 substitution release:

```bash
variantshift proteingym-download data/raw/proteingym --include-zero-shot-scores
```

This downloads the 43 MB assay archive, reference index, and approximately 1.9 GB official
zero-shot score archive. All remain under the ignored `data/raw/proteingym/` path; only aggregate
audits and metrics are committed.
