# Tiny outcome-blind example

This synthetic fixture exercises stable target/variant/prediction schemas, transport fitting,
content hashing, and the one-way confirmation lock without downloading biological measurements.

```bash
python examples/tiny/run_example.py --output-dir /tmp/variantshift-tiny
```

The run stops in `predictions_frozen`. That is intentional: a real confirmation must next publish
its preregistration, record the public URL, and reveal outcomes once. The example never fabricates
a registration or crosses that boundary.
