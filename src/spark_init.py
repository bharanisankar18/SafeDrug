import os
import sys
import ctypes
from pathlib import Path


def get_short_path(path_str: str) -> str:
    """Converts a Windows path to an 8.3 short path to avoid space issues in Spark/JVM."""
    if os.name == "nt" and os.path.exists(path_str):
        buf = ctypes.create_unicode_buffer(500)
        ctypes.windll.kernel32.GetShortPathNameW(path_str, buf, 500)
        return buf.value if buf.value else path_str
    return str(path_str)


# Project Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
HADOOP_HOME_DIR = BASE_DIR / "hadoop"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Configure Hadoop on Windows (winutils.exe and hadoop.dll)
if (HADOOP_HOME_DIR / "bin" / "winutils.exe").exists():
    short_hadoop = get_short_path(str(HADOOP_HOME_DIR)).replace("\\", "/")
    os.environ["HADOOP_HOME"] = short_hadoop
    os.environ["hadoop.home.dir"] = short_hadoop
    os.environ["PATH"] = str(HADOOP_HOME_DIR / "bin") + os.pathsep + os.environ.get("PATH", "")

# Configure exact Python executable for PySpark (8.3 short path without spaces)
short_python = get_short_path(sys.executable)
os.environ["PYSPARK_PYTHON"] = short_python
os.environ["PYSPARK_DRIVER_PYTHON"] = short_python

# Set JAVA_HOME if not already configured in environment
if "JAVA_HOME" not in os.environ or not os.path.exists(os.environ["JAVA_HOME"]):
    candidate_paths = [
        r"C:\Program Files\Java\jdk-24",
        r"C:\Program Files\Java\jdk-21",
        r"C:\Program Files\Java\jdk-17",
        r"C:\Program Files\Eclipse Adoptium\jdk-17",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            os.environ["JAVA_HOME"] = get_short_path(path)
            break

import pyspark
spark_home = Path(pyspark.__file__).resolve().parent
os.environ["SPARK_HOME"] = get_short_path(str(spark_home))

from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "DrugSafetySignalDetection", memory: str = "4g") -> SparkSession:
    """Builds and returns a tuned PySpark SparkSession for local execution."""
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[4]")
        .config("spark.driver.memory", memory)
        .config("spark.executor.memory", memory)
        .config("spark.network.timeout", "800s")
        .config("spark.executor.heartbeatInterval", "60s")
        .config("spark.sql.shuffle.partitions", "20")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.enabled", "false")
    )
    if (HADOOP_HOME_DIR / "bin" / "winutils.exe").exists():
        short_hadoop = get_short_path(str(HADOOP_HOME_DIR)).replace("\\", "/")
        builder = (
            builder
            .config("spark.hadoop.hadoop.home.dir", short_hadoop)
            .config("spark.driver.extraJavaOptions", f"-Dhadoop.home.dir={short_hadoop}")
            .config("spark.executor.extraJavaOptions", f"-Dhadoop.home.dir={short_hadoop}")
        )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


if __name__ == "__main__":
    print("--- SafeDose Step 1: Environment & Spark Verification ---")
    print(f"Python Executable: {os.environ.get('PYSPARK_PYTHON')}")
    print(f"JAVA_HOME:         {os.environ.get('JAVA_HOME')}")
    print(f"SPARK_HOME:        {os.environ.get('SPARK_HOME')}")
    print(f"Raw Data Dir:      {RAW_DATA_DIR}")
    print(f"Processed Dir:     {PROCESSED_DATA_DIR}")
    
    spark = get_spark_session("Step1-Verification")
    print(f"PySpark Version:   {spark.version}")
    
    df = spark.range(1, 11).toDF("number")
    row_count = df.count()
    print(f"Verified DataFrame range count: {row_count} rows")
    spark.stop()
    print(">>> Step 1 Complete: Spark is running smoothly! <<<")
