"""Spark session builder.

Reads master, driver/executor memory, shuffle partitions and S3A settings
from conf/<env>.yaml and supports three execution profiles:

* ``local``   - local[*] multi-threaded mode (host development)
* ``cluster`` - standalone Spark cluster (docker-compose spark-master/workers)
* ``k8s``     - Spark on Kubernetes (k8s:// master)

The profile is selected with the SPARK_PROFILE env var, an explicit argument,
or the ``spark.default_profile`` setting in the config file.
"""

from __future__ import annotations

import os
from typing import Any

from pyspark import SparkConf
from pyspark.sql import SparkSession

from common.config import load_config
from common.logging import get_logger

log = get_logger(__name__)

_DEFAULTS = {
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.session.timeZone": "UTC",
    "spark.ui.showConsoleProgress": "false",
}


def build_spark(
    app_name: str | None = None,
    conf_overrides: dict[str, str] | None = None,
    env: str | None = None,
    profile: str | None = None,
) -> SparkSession:
    """Build and return a configured SparkSession."""
    cfg = load_config(env)
    spark_cfg: dict[str, Any] = cfg.get("spark", {})
    app_name = app_name or spark_cfg.get("app_name") or "sparkfeaturestore"
    profile = profile or os.getenv("SPARK_PROFILE") or spark_cfg.get("default_profile", "local")

    profiles: dict[str, dict[str, Any]] = spark_cfg.get("profiles", {})
    if profile not in profiles:
        raise ValueError(f"Unknown Spark profile '{profile}'. Available: {sorted(profiles)}")
    prof = profiles[profile]

    conf = SparkConf()
    conf.setAppName(app_name)
    conf.setMaster(prof["master"])
    if "driver_memory" in prof:
        conf.set("spark.driver.memory", prof["driver_memory"])
    if "executor_memory" in prof:
        conf.set("spark.executor.memory", prof["executor_memory"])
    if "executor_cores" in prof:
        conf.set("spark.executor.cores", str(prof["executor_cores"]))
    if "driver_cores" in prof:
        conf.set("spark.driver.cores", str(prof["driver_cores"]))

    conf.set("spark.sql.shuffle.partitions", str(spark_cfg.get("shuffle_partitions", 200)))
    for key, value in _DEFAULTS.items():
        conf.set(key, value)
    for key, value in spark_cfg.get("extra", {}).items():
        conf.set(key, str(value))

    s3a: dict[str, Any] = cfg.get("s3a", {})
    if s3a.get("enabled", True):
        conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        if s3a.get("endpoint"):
            conf.set("spark.hadoop.fs.s3a.endpoint", s3a["endpoint"])
        if s3a.get("access_key"):
            conf.set("spark.hadoop.fs.s3a.access.key", s3a["access_key"])
        if s3a.get("secret_key"):
            conf.set("spark.hadoop.fs.s3a.secret.key", s3a["secret_key"])
        conf.set(
            "spark.hadoop.fs.s3a.path.style.access",
            str(s3a.get("path_style_access", True)).lower(),
        )
        conf.set(
            "spark.hadoop.fs.s3a.connection.ssl.enabled",
            str(s3a.get("ssl_enabled", False)).lower(),
        )

    for key, value in (conf_overrides or {}).items():
        conf.set(key, str(value))

    log.info(
        "Building SparkSession: app=%s profile=%s master=%s",
        app_name,
        profile,
        prof["master"],
    )
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    spark.sparkContext.setLogLevel(spark_cfg.get("log_level", "WARN"))
    return spark
