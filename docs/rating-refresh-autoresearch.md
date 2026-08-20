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
append input must expose the same file set, with the same byte counts and
SHA-256 values for every file. This is the fail-closed proof available to the
generic harness. A changed full Parquet file needs a richer row-level proof
before it can be used as an append-only phase. A removed game or a changed file
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
  "run": {
    "phase": "cold",
    "variant": "candidate",
    "entrypoint": "lol_kills.v2.tierlists.rating_refresh.refresh_ratings",
    "runtime_isolated": true,
    "accepted_census_bound": true,
    "timings": {
      "refresh_seconds": 12.345,
      "artifact_copy_hash_seconds": 0.456
    }
  },
  "outputs": {
    "player_ratings": {
      "path": "/path/to/player-ratings.json",
      "sha256": "...",
      "bytes": 123,
      "rows": 10
    },
    "team_ratings": {
      "path": "/path/to/team-ratings.json",
      "sha256": "...",
      "bytes": 456,
      "rows": 4
    }
  },
  "semantic": {
    "rating_schema": "..."
  }
}
```

The source binding must match the frozen fixture. Every output descriptor must
contain a real artifact path and a matching byte count. The runner hashes the
output descriptors without the path, so baseline and candidate paths may
differ. It compares every descriptor field and every semantic field exactly. A
different rating value, row count, digest, or source binding rejects the
candidate. The optional run section appears in the phase report and records the
entrypoint, isolation state, and adapter timings.

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

The repository adapter calls
`lol_kills.v2.tierlists.rating_refresh.refresh_ratings`. It stages the frozen
files under a private runtime directory and copies the production rating
manifest plus the complete stable refresh output inventory into the harness
output contract. The inventory is:

| Descriptor | Runtime file | Role |
| --- | --- | --- |
| `team_sequential` | `data/lol/features/ratings.parquet` | Sequential team rows |
| `team_dual_snapshot` | `data/lol/features/ratings_dual_snapshot.parquet` | Sequential team snapshot consumed by pack export |
| `team_snapshot` | `data/lol/features/ratings_snapshot.parquet` | Public team snapshot |
| `team_meta` | `data/lol/features/ratings_meta.json` | Team model metadata |
| `team_hierarchical_meta` | `data/lol/features/ratings_hierarchical_meta.json` | Hierarchical model metadata |
| `player_sequential` | `data/lol/features/player_ratings.parquet` | Sequential player rows |
| `player_snapshot` | `data/lol/features/player_ratings_snapshot.parquet` | Public player snapshot |
| `player_meta` | `data/lol/features/player_ratings_meta.json` | Player model metadata |
| `team_weekly` | `data/lol/features/team_weekly_ranks.json` | Team movement output |
| `player_weekly` | `data/lol/features/player_weekly_ranks.json` | Player movement output |

The adapter also records `rating_manifest` for
`data/lol/v2/tierlists/rating-refresh/rating-refresh-v1.json`. It fails when a
listed file is missing. It records the copied file size and SHA-256 digest.
The hierarchical cache snapshot, previous snapshot, cache manifest, and
player prefix cache remain private cache state. They are not output
descriptors. The harness gives baseline and candidate separate owner-marked
runtime roots. Each variant keeps its runtime from cold through append_only.
The active source files and accepted census are restaged for every phase.
Output artifacts go to a phase-specific directory.

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
  --baseline-command-json '["python3", "benchmarks/rating_refresh_adapter.py", "--min-games", "1", "--min-series", "1"]' \
  --candidate-command-json '["python3", "benchmarks/rating_refresh_adapter.py", "--min-games", "1", "--min-series", "1"]' \
  --budget-seconds 60 \
  --timeout-seconds 1800 \
  --report /private/tmp/rating-refresh-autoresearch/report.json
```

The report records both phases, both timings, return codes, timeout state,
stdout and stderr digests, call counts, output digests, and comparison reasons.
Each phase and variant also has bounded stdout and stderr log files under
`<output-root>/runs/<phase>/`. The report records each log path, stored-byte
digest, original byte count, and truncation flag. These logs keep adapter
failures inspectable without allowing unbounded process output into a report.
It records an invocation budget of four adapter calls. The process exits zero
only when both candidate phases satisfy the correctness gate and the 60-second
target. Add `--require-speedup` when an experiment must also be at least as
fast as its baseline in both phases.

The hard budget uses end-to-end adapter wall time. This includes subprocess
startup, input staging, the production refresh, artifact copying and hashing,
and adapter output writes. The repository adapter also reports the time spent
inside `refresh_ratings` and the time spent copying and hashing artifacts.
These sub-timings explain a wall-time result. They do not replace the budget
gate.

## Correctness gates

The comparison is accepted only when these conditions hold:

- the append census is a superset of the base census;
- every append input file has the same bytes and SHA-256 value as its base file;
- both adapters return successfully in each phase;
- each adapter binds to the phase count, identity digest, census digest, and
  frozen input-manifest digest;
- baseline and candidate output descriptors match exactly in each phase;
- baseline and candidate semantic outputs match exactly in each phase;
- every output descriptor has a real file path and matching byte count;
- the candidate phase stays at or below the 60-second budget.

The generic harness does not prove row-level append order inside a large
Parquet file. Use a new cold fixture when an existing source game changes. Keep
the append fixture for rows added after the accepted census only when the input
file bytes remain unchanged or a richer row-level proof is added.

## Local tests

The tests use two small JSONL fixtures and four short Python adapter calls.
They do not load the warehouse or run a rating refresh.

```sh
python3 -m pytest -q tests/test_rating_refresh_autoresearch.py
```
