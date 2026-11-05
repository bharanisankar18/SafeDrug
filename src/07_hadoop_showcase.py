"""
SafeDose - Hadoop Integration & Showcase Job
Demonstrates direct integration with Apache Hadoop 3.5.0:
1. Native Hadoop FileSystem API (org.apache.hadoop.fs.FileSystem, Path, FileStatus)
   - Inspects storage paths, block sizes, replication factors, and file lengths.
2. Classic Hadoop MapReduce Workflow (Map-Shuffle-Sort-Reduce)
   - Executes standard Mapper and Reducer functions over FAERS adverse event records.
   - Saves results to data/processed/hadoop_mapreduce_counts/
"""

import sys
import time
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spark_init import PROCESSED_DATA_DIR, RAW_DATA_DIR, get_spark_session


def run_hadoop_showcase():
    start_time = time.time()
    print("=" * 70)
    print("  SAFEDOSE - APACHE HADOOP INTEGRATION & MAPREDUCE SHOWCASE")
    print("=" * 70)

    spark = get_spark_session(app_name="SafeDose_HadoopShowcase", memory="4g")
    sc = spark.sparkContext
    jvm = spark._jvm

    # -------------------------------------------------------------
    # 1. Inspect Native Apache Hadoop Environment
    # -------------------------------------------------------------
    hadoop_version = jvm.org.apache.hadoop.util.VersionInfo.getVersion()
    hadoop_build = jvm.org.apache.hadoop.util.VersionInfo.getDate()
    hadoop_conf = sc._jsc.hadoopConfiguration()

    print(f"\n[1] Native Apache Hadoop Environment:")
    print(f"    • Hadoop Version        : {hadoop_version}")
    print(f"    • Hadoop Build Date     : {hadoop_build}")
    print(f"    • Default FileSystem    : {hadoop_conf.get('fs.defaultFS')}")
    print(f"    • Hadoop OutputCommitter: {hadoop_conf.get('spark.sql.sources.commitProtocolClass', 'Hadoop FileOutputCommitter')}")

    # -------------------------------------------------------------
    # 2. Native Hadoop FileSystem API (org.apache.hadoop.fs.FileSystem)
    # -------------------------------------------------------------
    print(f"\n[2] Interacting with Hadoop FileSystem API (org.apache.hadoop.fs.FileSystem):")
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    print(f"    • Active FS Implementation: {fs.getClass().getName()}")
    print(f"    • FS Scheme               : {fs.getScheme()}")

    processed_hadoop_path = jvm.org.apache.hadoop.fs.Path(str(PROCESSED_DATA_DIR.as_posix()))
    file_statuses = fs.listStatus(processed_hadoop_path)

    print(f"\n    Hadoop File System Directory Inspection ({processed_hadoop_path}):")
    print(f"    {'-'*65}")
    print(f"    {'File Name':<32} {'Size (KB)':<12} {'Block Size':<12} {'Replication'}")
    print(f"    {'-'*65}")
    for status in file_statuses:
        name = status.getPath().getName()
        size_kb = status.getLen() / 1024.0
        block_size = status.getBlockSize() / (1024.0 * 1024.0) # in MB
        rep = status.getReplication()
        print(f"    {name:<32} {size_kb:>10.1f} KB {block_size:>10.0f} MB {rep:>8}")

    # -------------------------------------------------------------
    # 3. Classic Hadoop MapReduce Workflow on Adverse Drug Events
    # -------------------------------------------------------------
    print(f"\n[3] Executing Classic Hadoop MapReduce Workflow:")
    print(f"    • Mapping stage  : Extract (drug_name, 1) and (reaction_name, 1)")
    print(f"    • Shuffle & Sort : Partitioned grouping by key across workers")
    print(f"    • Reducing stage : Summing counts with reduceByKey()")

    clean_parquet_path = PROCESSED_DATA_DIR / "clean_reports.parquet"
    if not clean_parquet_path.exists():
        raise FileNotFoundError(f"Missing {clean_parquet_path}. Run Step 3 first.")

    df = spark.read.parquet(str(clean_parquet_path.as_posix()))

    # Map phase: RDD key-value pairs
    rdd = df.select("drug_name", "reaction_name").rdd

    # Drug frequency MapReduce
    # Mapper: record -> (drug, 1)
    # Reducer: (c1, c2) -> c1 + c2
    drug_mr_counts = (
        rdd.map(lambda row: (row["drug_name"], 1))
        .reduceByKey(lambda a, b: a + b)
        .sortBy(lambda pair: pair[1], ascending=False)
    )

    # Reaction frequency MapReduce
    reaction_mr_counts = (
        rdd.map(lambda row: (row["reaction_name"], 1))
        .reduceByKey(lambda a, b: a + b)
        .sortBy(lambda pair: pair[1], ascending=False)
    )

    top_drugs = drug_mr_counts.take(5)
    top_reactions = reaction_mr_counts.take(5)

    print("\n    Hadoop MapReduce Output - Top 5 Drugs by Frequency:")
    for rank, (drug, count) in enumerate(top_drugs, 1):
        print(f"      {rank}. {drug:<20} -> {count:,} reports")

    print("\n    Hadoop MapReduce Output - Top 5 Reactions by Frequency:")
    for rank, (rx, count) in enumerate(top_reactions, 1):
        print(f"      {rank}. {rx:<20} -> {count:,} reports")

    # -------------------------------------------------------------
    # 4. Save MapReduce Output using Hadoop Output Formatting
    # -------------------------------------------------------------
    mr_output_dir = PROCESSED_DATA_DIR / "hadoop_mr_counts"
    print(f"\n[4] Saving Hadoop MapReduce Counts to {mr_output_dir}...")

    # Convert to DataFrame and write to Parquet
    mr_df = (
        drug_mr_counts.map(lambda p: (p[0], p[1]))
        .toDF(["item_name", "event_count"])
    )
    
    if mr_output_dir.exists():
        shutil.rmtree(mr_output_dir, ignore_errors=True)

    try:
        mr_df.coalesce(1).write.mode("overwrite").parquet(str(mr_output_dir.as_posix()))
    except Exception as err:
        pdf = mr_df.toPandas()
        pdf.to_parquet(mr_output_dir, engine="pyarrow", index=False)

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Hadoop integration showcase completed successfully in {elapsed:.1f}s.")
    print("=" * 70)

    spark.stop()


if __name__ == "__main__":
    run_hadoop_showcase()

