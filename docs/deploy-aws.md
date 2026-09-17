# Deploying on AWS

What changes between the local kind proof and AWS: **the config profile and
the secret source. Nothing else.** The same images
(`sparkfeaturestore/spark:3.5.9`, `sparkfeaturestore/train:0.1.0`), the same
spark-submit Job manifests and the same code run unchanged; S3A and the
Postgres registry are addressed entirely through `conf/k8s.yaml`.

## Proven claim

The kind proof ran the full pipeline end-to-end with:

* `conf/k8s.yaml` pointing S3A at `minio.spark-jobs.svc.cluster.local:9000`
  with dev access keys, and Postgres at the in-cluster StatefulSet;
* dev credentials from `k8s/secrets.example.yaml` (substituted by make).

On AWS the ONLY differences are:

1. `conf/k8s.yaml`: remove `s3a.endpoint` (empty endpoint = real S3) and
   remove `s3a.access_key` / `s3a.secret_key` (pod identity via IRSA).
   `build_spark` only sets the keys it finds, so the AWS credential chain
   (IRSA web identity token) is used automatically. Same for the training
   `_read_table`: no endpoint -> default AWS S3 filesystem + IRSA.
2. `postgres.host` points at the RDS endpoint, and the password comes from a
   real secret (SealedSecrets / ExternalSecrets / AWS Secrets Manager)
   instead of the dev `sparkfeaturestore-secrets` Secret.

No image rebuilds, no manifest changes, no code changes.

## S3 bucket layout

One bucket per environment (`sparkfeaturestore-datalake-<env>`):

```text
s3://sparkfeaturestore-datalake-prod/
├── raw/                     # fetched NYC TLC parquet (scripts/fetch_data.py)
├── bronze/trips/            # event_date=YYYY-MM-DD/ partitions
├── silver/
│   ├── pickup_zone_demand/
│   ├── driver_trip_history/
│   └── trip_features/
└── models/                  # optional model artifacts (elsewhere today)
```

The prefix is the only bucket-specific bit: `conf/k8s.yaml` paths are
`s3a://sparkfeaturestore-datalake-prod/{raw,bronze,silver,...}`.

Notes:

* Enable versioning + lifecycle rules on `raw/` and `bronze/`; expire
  staging prefixes (`*.staging/*`) after 7 days.
* Block public access; access is pod-level only (see IRSA below).

## IRSA (IAM Roles for Service Accounts)

Pod-level S3 auth, no long-lived keys:

```bash
eksctl create iamserviceaccount \
  --cluster <cluster> \
  --namespace spark-jobs \
  --name spark \
  --role-name sparkfeaturestore-s3 \
  --attach-policy-arn arn:aws:iam::<acct>:policy/SparkFeatureStoreDatalake \
  --approve
```

```json
// SparkFeatureStoreDatalake policy
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::sparkfeaturestore-datalake-prod",
        "arn:aws:s3:::sparkfeaturestore-datalake-prod/*"
      ]
    }
  ]
}
```

Annotate the ServiceAccount in `k8s/rbac.yaml` with the role ARN:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: spark
  namespace: spark-jobs
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::<acct>:role/sparkfeaturestore-s3
```

With the endpoint and keys removed from `conf/k8s.yaml`, `hadoop-aws` picks
up the web identity token automatically; the training jobs use pyarrow's
default credential chain, which also honors IRSA.

## EKS node sizing

The driver+executor profile that worked on kind (4 cores, 1g executors) is
deliberately small. For the full 2023 dataset (~40M rows) on EKS:

| Component | Instance | Rationale |
|---|---|---|
| node group | m6i.4xlarge (16 vCPU / 64 GiB) | 2 executors per node, shuffle + window buffers in RAM |
| driver pod | 2 vCPU / 4 GiB request | client aggregations, registry, quality checks |
| executor pod | 4 vCPU / 12 GiB request | trailing-window sort per zone; trip_features hash join |
| executor count | 4-8 (`EXECUTOR_INSTANCES` in the Job manifests / `make k8s-features EXECUTOR_INSTANCES=6`) | linear in dataset size |
| shuffle partitions | 200-400 (`conf/k8s.yaml` `shuffle_partitions`) | re-run `bench/partition_sweep.py` on the target topology; on the 4-core kind cluster AQE-off + >=200 minimized stragglers |
| training job | 2 vCPU / 6-8 GiB | HistGB fits 40M rows in ~8 GiB with native categoricals (no OHE expansion) |

## Sizing knobs that are already in place

* `k8s/spark-submit-{bronze,features}.yaml` -- `EXECUTOR_INSTANCES`,
  `EXECUTOR_MEMORY` (substituted by make, e.g.
  `make k8s-features EXECUTOR_INSTANCES=6 EXECUTOR_MEMORY=12g`).
* `conf/k8s.yaml` -- `shuffle_partitions`, driver/executor memory profile.
* `k8s/train-job.yaml` -- CPU/memory requests and limits.

## Migration checklist

1. Create the bucket + IRSA role (above); set the bucket prefix in `conf/k8s.yaml`.
2. RDS Postgres 16 (`feature_store` db, `sparkfs` user); update
   `postgres.host` + the password secret reference.
3. Push `sparkfeaturestore/spark:3.5.9` + `train:0.1.0` to ECR; update
   `spark.kubernetes.container.image` and the Job/CronJob image fields
   (they are plain text substitutions).
4. `kubectl apply -f k8s/{namespace,rbac,configmap}.yaml` + the real secret.
5. Seed `s3://.../raw/` (upload the fetched parquet).
6. `kubectl apply -f k8s/spark-submit-bronze.yaml`,
   `k8s/spark-submit-features.yaml`, `k8s/cron-features.yaml`,
   `k8s/train-job.yaml`.
7. Verify: bronze row counts in Postgres `feature_runs`, silver partitions
   in S3, a training run registered in `model_runs`.

## What the CronJob gives you

`k8s/cron-features.yaml` schedules the silver build daily (06:00),
`concurrencyPolicy: Forbid`, and Spark cleans up stale driver pods on
resubmission. Failures are visible as failed driver pods + `failed` rows in
`feature_runs`; a successful re-run supersedes nothing (run_key no-op logic
still applies via the registry).
