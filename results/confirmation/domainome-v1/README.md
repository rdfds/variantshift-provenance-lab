# Domainome target records

This directory contains the frozen target protocol and outcome lock for the Domainome panel.

## Source receipt location

The source receipt belongs at this path relative to the repository root:

```text
data/confirmation-targets/domainome-v1/target-extraction-receipt.json
```

The frozen protocol and lock record an absolute path from the original environment. Its
portable equivalent is the path above. The receipt is not included in the public checkout;
a recovered copy must match SHA-256
`da0e388f36f6b2bf9be8a96e0f4be7745a97125ba24b22770bdf6cc762f73984`.
A newly generated receipt is a new artifact and may have a different digest.

The protocol and lock are retained unchanged because their hashes record the original freeze.
This location note does not change lock resolution or authorize outcome access. New panel
freezes use working-directory-relative paths for source artifacts inside that directory.
Run from the repository root with a relative output directory to keep those paths portable.
