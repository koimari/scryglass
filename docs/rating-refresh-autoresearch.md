# Rating refresh autoresearch

This harness measures a rating refresh against one frozen input package. It
uses four adapter calls:

1. baseline with the cold census
2. candidate with the cold census
3. baseline with the append-only census
4. candidate with the append-only census

The target is 60 seconds per candidate phase. The harness reports wall time in
seconds and milliseconds. It reports adapter call counts from a JSON file or
from one-line call markers. The harness does not modify the production worker.

## Freeze the input

Use a source directory that contains the input files. Pass the same relative
file list for the base and append snapshots. Pass the accepted census for each
snapshot. Use a new or empty output directory for each freeze.

```sh
python3 benchmarks/rating_refresh_autoresearch.py \
  --base-root /path/to/base-source \
  --base-census /path/to/base-accepted-census.json \
  --append-root /path/to/append-source \
  --append-census /path/to/append-accepted-census.json \
  --input-relative data/lol/warehouse/parquet/oe_live/maps.parquet \
  --input-relative data/lol/warehouse/parquet/oe_live/oe_player_games.parquet \
  --input-relative data/lol/warehouse/parquet/oe_live/oe_team_games.parquet \
  --output-root /private/tmp/rating-refresh-autoresearch \
  --freeze-only
```

The freeze copies every input and both census files. The freeze manifest stores
file byte counts and SHA-256 digests. It stores the census game count and
source identity digest. The append census must contain every base game ID. The
append input must expose the same file set. A removed game or a changed file
set blocks the benchmark.

The harness creates these paths:

```text
<output-root>/freeze.json
<output-root>/frozen/cold/manifest.json
<output-root>/frozen/cold/accepted-census.json
<output-root>/frozen/cold/inputs/...
<output-root>/frozen/append_only/manifest.json
<output-root>/frozen/append_only/accepted-census.json
<output-root>/frozen/append_only/inputs/...
```

The source root is not used after the copy. This gives each experiment a
stable input path.

## Adapter contract

The baseline and candidate commands are JSON argv arrays. The runner passes
the fixture through environment variables:

```text
SCRYGLASS_RATING_AUTORESEARCH_INPUT_ROOT
SCRYGLASS_RATING_AUTORESEARCH_CENSUS_PATH
SCRYGLASS_RATING_AUTORESEARCH_FIXTURE_MANIFEST
SCRYGLASS_RATING_AUTORESEARCH_FIXTURE_MANIFEST_SHA256
SCRYGLASS_RATING_AUTORESEARCH_OUTPUT_MANIFEST
SCRYGLASS_RATING_AUTORESEARCH_CALL_COUNTS_PATH
SCRYGLASS_RATING_AUTORESEARCH_PHASE
SCRYGLASS_RATING_AUTORESEARCH_VARIANT
```

The adapter writes one JSON output manifest to the path in
`SCRYGLASS_RATING_AUTORESEARCH_OUTPUT_MANIFEST`.

```json
{
  "schema_version": "scryglass:rating-autoresearch-output:v1",
  "source": {
    "phase": "cold",
    "source_game_count": 17762,
    "source_identity_sha256": "...",
    "census_sha256": "...",
    "input_manifest_sha256": "..."
  },
  "outputs": {
    "player_ratings": {
      "path": "/path/to/player-ratings.json",
      "sha256": "...",
      "bytes": 123,
      "rows": 10
    },
    "team_ratings": {
      "sha256": "...",
      "rows": 4
    }
  },
  "semantic": {
    "rating_schema": "..."
  }
}
```

The source binding must match the frozen fixture. The runner checks an
artifact path when one is provided. It hashes the output descriptors without
the path, so baseline and candidate paths may differ. It compares every
descriptor field and every semantic field exactly. A different rating value,
row count, digest, or source binding rejects the candidate.

The adapter writes call counts as either a mapping or an object with a
`counts` mapping:

```json
{"counts": {"player_elo": 8, "global_player_bt": 16}}
```

Every count must be a non-negative integer. If the file is absent, the runner
looks for one marker per call in stdout or stderr:

```text
[rating-autoresearch] call name=player_elo
```

An absent count source is reported as `unavailable`. The timing remains valid,
but a release decision should require the counts needed for the experiment.

## Run the fixed comparison

Use an adapter for each implementation. The command receives the fixture
through the environment. The shell is not used.

```sh
python3 benchmarks/rating_refresh_autoresearch.py \
  --base-root /path/to/base-source \
  --base-census /path/to/base-accepted-census.json \
  --append-root /path/to/append-source \
  --append-census /path/to/append-accepted-census.json \
  --input-relative data/lol/warehouse/parquet/oe_live/maps.parquet \
  --input-relative data/lol/warehouse/parquet/oe_live/oe_player_games.parquet \
  --input-relative data/lol/warehouse/parquet/oe_live/oe_team_games.parquet \
  --output-root /private/tmp/rating-refresh-autoresearch \
  --baseline-command-json '["python3", "path/to/rating_adapter.py"]' \
  --candidate-command-json '["python3", "path/to/rating_adapter.py"]' \
  --budget-seconds 60 \
  --timeout-seconds 1800 \
  --report /private/tmp/rating-refresh-autoresearch/report.json
```

The report records both phases, both timings, return codes, timeout state,
stdout and stderr digests, call counts, output digests, and comparison reasons.
It records an invocation budget of four adapter calls. The process exits zero
only when both candidate phases satisfy the correctness gate and the 60-second
target. Add `--require-speedup` when an experiment must also be at least as
fast as its baseline in both phases.

## Correctness gates

The comparison is accepted only when these conditions hold:

- the append census is a superset of the base census;
- both adapters return successfully in each phase;
- each adapter binds to the phase count, identity digest, census digest, and
  frozen input-manifest digest;
- baseline and candidate output descriptors match exactly in each phase;
- baseline and candidate semantic outputs match exactly in each phase;
- the candidate phase stays at or below the 60-second budget.

The census check cannot identify an in-place correction inside a large parquet
file. Build a new cold fixture when an existing source game changes. Keep the
append fixture for rows added after the accepted census. Preserve the source
receipt and the output manifests with the report.

## Local tests

The tests use two small JSONL fixtures and four short Python adapter calls.
They do not load the warehouse or run a rating refresh.

```sh
python3 -m pytest -q tests/test_rating_refresh_autoresearch.py
```
