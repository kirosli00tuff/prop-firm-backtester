# Benchmarks

Wall time and peak resident memory of the reference strategy on NQ 1-minute bars over three open windows (one, five and ten years)
(allowlisted engine checks E6_1Y, E6_5Y, E6_10Y in config/engine_checks.toml). Single process, one run at a time.

## Machine (generic)

- A laptop class machine: about 20 threads, 16 GB RAM, a recent Linux kernel
- CPython 3.12 (project interpreter, run through uv)
- Storage holding the bar stores: a separate local disk, read in place

## Result

| window | wall cold (s) | wall warm (s) | peak RSS (MB) |
|--------|--------------:|--------------:|--------------:|
| E6_1Y (one year) | 6.89 | 6.81 | 438 |
| E6_5Y (five years) | 18.57 | 19.46 | 1,139 |
| E6_10Y (ten years) | 31.89 | 31.80 | 2,071 |

Notes
- Wall time is `/usr/bin/time -v` elapsed time of the whole command, so it includes
  interpreter and uv start-up (a fixed cost of a few seconds that weighs most on the 1Y row).
- Peak RSS is the larger of the cold and warm run (maximum resident set size, converted at 1024 kB per MB).
- Cold and warm differ by under 5 percent on every window. The page cache cannot be dropped without root, so the
  "cold" run is the first run of that window in the benchmark session, not a cold-disk run.
- Each run appended one engine_check record and N did not change.
- The runs used licensed bars that the public copy does not ship; the figures are timings only.

## Command

```
/usr/bin/time -v uv run pfb run strategies/reference_breakout.py --kind engine_check --check-id <id> --json
```

with `<id>` one of E6_1Y, E6_5Y, E6_10Y, run twice in a row per window from the repository root.
