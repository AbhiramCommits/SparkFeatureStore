"""Pytest fixtures and environment tuning for Spark tests.

Local PySpark needs a JDK 8/11/17 (Spark 3.5 does not support Java 21+).
This module auto-selects a compatible JDK (via /usr/libexec/java_home on
macOS) and un-sets a SPARK_HOME whose bundled Spark version does not match
the installed PySpark package (e.g. Homebrew's Spark 4.x). Spark tests are
skipped when no compatible runtime is available.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

SUPPORTED_JAVA_MAJORS = {8, 11, 17}


def _java_major(version_string: str) -> int:
    parts = version_string.strip().split(".")
    return int(parts[1]) if parts[0] == "1" else int(parts[0])


def _major_of_java_home(java_home: str) -> int | None:
    java_bin = Path(java_home) / "bin" / "java"
    if not java_bin.exists():
        return None
    try:
        proc = subprocess.run(
            [str(java_bin), "-version"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    m = re.search(r'version "([^"]+)"', proc.stderr or proc.stdout)
    return _java_major(m.group(1)) if m else None


def _find_compatible_java_home() -> str | None:
    if "JAVA_HOME" in os.environ:
        major = _major_of_java_home(os.environ["JAVA_HOME"])
        if major in SUPPORTED_JAVA_MAJORS:
            return os.environ["JAVA_HOME"]
    try:
        proc = subprocess.run(
            ["/usr/libexec/java_home", "-V"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return None
    for line in proc.stderr.splitlines():
        m = re.match(r"\s*(\d\S+)\s", line)
        if not m:
            continue
        if _java_major(m.group(1)) not in SUPPORTED_JAVA_MAJORS:
            continue
        for token in line.split():
            if token.startswith("/") and Path(token).exists():
                return token
    return None


def _fix_spark_home() -> None:
    """Drop SPARK_HOME if it bundles a different Spark version than PySpark."""
    spark_home = os.environ.get("SPARK_HOME")
    if not spark_home:
        return
    try:
        import pyspark
    except ImportError:
        return
    jars = list(Path(spark_home).glob("jars/spark-core_*.jar"))
    if not jars:
        return
    m = re.search(r"spark-core_[\d.]+-([\d.]+)\.jar$", jars[0].name)
    if m and m.group(1) != pyspark.__version__:
        os.environ.pop("SPARK_HOME", None)


def pytest_configure(config) -> None:  # noqa: ARG001
    java_home = _find_compatible_java_home()
    if java_home:
        os.environ["JAVA_HOME"] = java_home
    _fix_spark_home()


@pytest.fixture(scope="session")
def spark():
    """Session-scoped local SparkSession; skipped if no compatible JDK."""
    try:
        from pyspark.sql import SparkSession

        session = (
            SparkSession.builder.master("local[2]")
            .appName("pytest")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.driver.host", "127.0.0.1")
            .getOrCreate()
        )
        session.sparkContext.setLogLevel("ERROR")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Spark local session unavailable: {exc}")
    yield session
    session.stop()
