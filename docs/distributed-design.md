# Distributed Design

How the pipeline behaves as distributed infrastructure: partitioning, shuffle
behavior, skew handling, failure modes and the consistency guarantees of the
write path. All numbers below are measured on the docker-compose cluster
(1 master, 2 workers x 2 cores / 3 GB, 4 executors total) over the Jan-2023
Yellow Taxi dataset: **3,066,766 trips / 36 `event_date` partitions**.

---

## 1. Partitioning strategy

| Layer | Layout | Partition key | Rationale |
|---|---|---|---|
| raw | `data/raw/*.parquet` | none (monthly files) | source of truth, append-only |
| bronze | `bronze/trips/event_date=YYYY-MM-DD/` | `event_date` (day) | date-range scans; 36 small dirs/month |
| silver | `silver/<group>/event_date=YYYY-MM-DD/` | `event_date` (day) | aligned with bronze; training slices by date |

Rules:

* Hive-style `key=value` partitions only; readers use partition discovery.
* Re-processing uses **full-table replacement** with a single atomic swap
  (see §4) — there are no partial partition sets.
* `shuffle_partitions` defaults to 16 on the small cluster and 64 on k8s
  (`conf/*.yaml`); see the sweep results below for tuning.

## 2. Shuffle behavior

Where real shuffles happen:

* **bronze**: none (single-pass transform, partition on write).
* **pickup_zone_demand**: `Window.partitionBy(pu_location_id).orderBy(unix_ts)`
  with `rangeBetween(-w, -1)` — a full-row shuffle by zone (260 keys).
* **driver_trip_history**: pre-aggregated to (vendor, day) partials first
  (~730 rows), then range windows — shuffle is negligible.
* **trip_features**: bucket-probed hash joins on `(key, ts_bucket)` — the
  `as_of_join` implementation joins on `(keys, bucket)` with an equi-join
  (bucket size = tolerance + 1) instead of a per-key cartesian; each spine
  row probes its own bucket and the previous one.

Measured shuffle sizes (window workload, 3M rows):

```text
window shuffle write/read    ~16-19 MB (all rows, 4-5 columns)
groupBy(key) partials        ~13 KB   (map-side combine: count/sum/avg)
full-row keyed repartition   ~7.4 MB  (16 partitions)
```

### Partition sweep (measured)

Workload: `pickup_zone_demand` (1h/6h/24h trailing windows) + aggregation
over the window columns, `make bench-partitions`, CSV in
`bench/results/partition_sweep.csv`:

| shuffle.partitions | AQE | wall (s) | max task (ms) | shuffle read (MB) |
|---|---|---|---|---|
| 50   | on  | 25.14 | 21,360 | 19.2 |
| 50   | off | 26.32 | 6,743  | 19.2 |
| 200  | on  | 24.88 | 21,314 | 16.8 |
| 200  | off | 25.58 | 2,984  | 16.8 |
| 800  | on  | 26.92 | 22,661 | 16.2 |
| 800  | off | 37.86 | 1,588  | 16.3 |
| 2000 | on  | 23.32 | 18,437 | 16.0 |
| 2000 | off | 26.07 | 2,124  | 16.1 |

Observations:

* Wall time is flat (~23-27s) — the 4-core cluster is CPU-bound; total work
  dominates, not the shuffle.
* **AQE ON creates a straggler**: `CoalesceShufflePartitions` merges post-
  shuffle partitions and one task ends up holding the hottest zone
  (~160k rows x 3 range windows -> 18-23s max task). AQE OFF keeps tasks
  small (1.6-6.7s max) at the cost of more, smaller tasks.
* Small clusters: prefer **AQE off + shuffle.partitions >= 200** to avoid
  stragglers. Large clusters: re-run the sweep; AQE + higher partitions
  typically wins when the cluster has enough cores to hide the skew.

## 3. Skew

`PULocationID` is long-tailed (24 zones above 50k trips/month; the busiest
zone, 132, holds ~5% of rows in Jan). `features/skew.py`:

* `detect_hot_keys(df, key_col, threshold)` — runtime approximate frequency
  count (sampled groupBy, scaled); only keys above the threshold are salted.
* `add_salt(df, hot_col, hot_keys, salt_buckets)` — hot rows get a random
  salt in `[0, salt_buckets)`; cold keys stay unsalted.
* `salted_aggregate(...)` — two-phase: partial aggregates over
  `(key, salt)`, then de-salted combination (count = sum of partial counts,
  avg = sum of sums / sum of counts). Result equals the unsalted result
  (verified in the bench: all 257 keys exact within float ULP).

Measured (`make bench-skew`, CSV in `bench/results/skew_bench.csv`):

| mode | wall (s) | max task (ms) | shuffle read (MB) |
|---|---|---|---|
| unsalted groupBy | 0.67 | 290 | 0.013 |
| salted groupBy | 0.49 | 274 | 0.013 |
| unsalted full-row shuffle | 0.87 | 395 | 7.4 |
| salted full-row shuffle (2 shuffles) | 1.08 | 375 | 17.7 |

Honest conclusion for this dataset size: the hottest zone is only ~5% of
rows, so the unsalted shuffle is barely skewed (max-task skew ~1.2x) and
salting **adds** cost (a second full shuffle). Salting wins when hot keys
reach ~20-30% of rows (e.g. multi-year data, or when the skew column has
few dominant values) — the threshold is configurable, so the pipeline
measures skew at runtime and only salts when it actually helps.

## 4. Consistency: staging + atomic promotion

Every table write goes through `common/atomic.py`:

```text
1. write -> <table>.staging/<run_id>/          (sibling dir, invisible to readers)
2. existing <table> -> <table>.trash/          (only if present)
3. rename <table>.staging/<run_id> -> <table>  (single atomic rename, POSIX)
4. write <table>/_SUCCESS
5. delete <table>.trash
```

What is atomic and what is not (be precise):

* **Atomic on POSIX/HDFS**: step 3 is a single `rename` on the same
  filesystem. Readers of the final path see either the complete old table
  or the complete new table — never a mix, never a half-written partition.
* **Rollback**: if step 3 or 4 fails, the old table is renamed back from
  `.trash` (best effort). A crash that skips rollback leaves the old table
  in `<table>.trash/` and NO final table — the next run's staging cleanup
  removes the leftovers and re-promotes.
* **NOT atomic on S3A**: `rename` is implemented as copy + delete. Readers
  must gate on the `_SUCCESS` marker (written only after the swap). The
  pipeline is local-FS/MinIO-local today; k8s/S3 readers should wait for
  `_SUCCESS`.
* **Not atomic**: the Postgres `feature_runs` row and the table swap are
  two separate systems (no 2PC). A crash between write and registry
  `finish_run` leaves a `running` row; the next run sees no completed
  `run_key` and rebuilds.

## 5. Idempotency + retry

* **run_key** = sha256(feature_group + git_sha + sorted input paths). The
  Postgres `feature_runs` table has a partial unique index:
  `UNIQUE (run_key) WHERE status = 'success'` — at most one successful run
  per (group, code, inputs).
* Re-running a completed run_key is a **no-op** unless `--force` (which
  marks the previous success `superseded` and rebuilds). Failed runs do not
  block retries. Verified: a second `make silver` skips all three groups in
  ~0s with no Spark work.
* `common/retry.py`: exponential backoff with full jitter, bounded attempts.
  Only transient errors are retried (S3 5xx, 429, connection reset, timeout,
  broken pipe — classified by message/type); data-quality failures
  (`DataQualityError`) are never retried.
* Kill-mid-write is tested (`tests/test_atomic.py`): an injected exception
  during promotion triggers rollback; a re-run produces a consistent table
  with no duplicate or partial partitions and no staging leftovers.

## 6. Failure modes

| Failure | Effect | Recovery |
|---|---|---|
| executor OOM during windows | job fails, registry row `failed` | re-run; completed run_keys unchanged; driver history pre-rollup avoids vendor skew OOM |
| crash mid-write | staged data orphaned, final table untouched | next run cleans stale staging + re-promotes |
| crash mid-promotion | old table in `.trash`, final absent or old | rollback on error path; re-run restores |
| Postgres down at start | connect retried with backoff | `--no-registry` to bypass |
| S3 5xx on read | retry with backoff | permanent data-quality errors abort immediately |

## 7. What is NOT covered (yet)

* Incremental/merge writes: `atomic_write_parquet` implements overwrite
  semantics (full-table swap). Incremental partition appends would need a
  per-partition promote path.
* S3A rename is not atomic (see §4); production S3 readers must use the
  `_SUCCESS` gate.
* Cross-system consistency between the table swap and `feature_runs` (no
  transaction) — the registry is a best-effort metadata log.
