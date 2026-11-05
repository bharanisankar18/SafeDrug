"""
SafeDose - Step 6: Time Trends & Safety Spike Detection
Analyzes monthly adverse event report volumes per drug and per signal pair over time.
Computes historical baseline mean, standard deviation, and Z-scores to flag sudden
safety signal surges (spikes with Z >= 3.0 and count >= 3).

Input:
  - data/processed/clean_reports.parquet
  - data/processed/signals.parquet
Output:
  - data/processed/trends.parquet
"""

import sys
import time
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spark_init import PROCESSED_DATA_DIR, get_spark_session
from pyspark.sql.functions import (
    col, substring, concat, lit, to_date, countDistinct,
    avg, stddev, coalesce, when, round as spark_round, broadcast
)
from pyspark.sql.window import Window


def compute_safety_trends():
    start_time = time.time()
    print("=== SafeDose Step 6: Adverse Event Trends & Spike Detection Job ===")

    clean_path = PROCESSED_DATA_DIR / "clean_reports.parquet"
    signals_path = PROCESSED_DATA_DIR / "signals.parquet"
    output_path = PROCESSED_DATA_DIR / "trends.parquet"

    if not clean_path.exists():
        raise FileNotFoundError(f"Missing required input: {clean_path}. Run Step 3 first.")
    if not signals_path.exists():
        raise FileNotFoundError(f"Missing required input: {signals_path}. Run Step 4 first.")

    spark = get_spark_session(app_name="SafeDose_TrendsAnalysis", memory="4g")

    print(f"Loading clean reports from {clean_path}...")
    df = spark.read.parquet(str(clean_path.as_posix()))

    print(f"Loading safety signals from {signals_path}...")
    signals_df = spark.read.parquet(str(signals_path.as_posix()))
    active_signals = signals_df.filter(col("is_signal") == True).select("drug_name", "reaction_name").distinct()
    num_signals = active_signals.count()
    print(f"Active safety signals for trend tracking: {num_signals:,}")

    # Standardize date and extract Year-Month
    # FAERS receivedate is in YYYYMMDD format
    df_dated = (
        df.filter(col("receivedate").isNotNull() & (col("receivedate") != ""))
        .withColumn("year_month", concat(substring(col("receivedate"), 1, 4), lit("-"), substring(col("receivedate"), 5, 2)))
        .withColumn("date", to_date(concat(col("year_month"), lit("-01"))))
    )

    print("Computing drug-level monthly report volume...")
    # 1. Drug-level overall trends (reaction_name = 'ALL')
    drug_monthly = (
        df_dated.groupBy("drug_name", "year_month", "date")
        .agg(countDistinct("safetyreportid").alias("report_count"))
        .withColumn("reaction_name", lit("ALL"))
    )

    print("Computing signal-pair monthly report volume...")
    # 2. Specific drug-reaction signal trends
    sig_monthly = (
        df_dated.join(broadcast(active_signals), on=["drug_name", "reaction_name"], how="inner")
        .groupBy("drug_name", "reaction_name", "year_month", "date")
        .agg(countDistinct("safetyreportid").alias("report_count"))
    )

    # 3. Union both levels
    combined_monthly = drug_monthly.unionByName(sig_monthly)

    print("Calculating rolling statistical baselines and Z-score spikes...")
    # Baseline mean and standard deviation per (drug_name, reaction_name)
    w = Window.partitionBy("drug_name", "reaction_name")
    trends_df = (
        combined_monthly
        .withColumn("mean_count", spark_round(avg("report_count").over(w), 2))
        .withColumn("std_count", spark_round(coalesce(stddev("report_count").over(w), lit(0.0)), 2))
        .withColumn(
            "z_score",
            spark_round(
                when(col("std_count") > 0.0, (col("report_count") - col("mean_count")) / col("std_count"))
                .otherwise(lit(0.0)),
                2
            )
        )
        # Flag spikes: Z-score >= 3.0 and at least 3 reports in that month
        .withColumn(
            "is_spike",
            (col("z_score") >= 3.0) & (col("report_count") >= 3)
        )
    )

    total_records = trends_df.count()
    spike_records = trends_df.filter(col("is_spike") == True).count()
    drug_spikes = trends_df.filter((col("is_spike") == True) & (col("reaction_name") == "ALL")).count()
    signal_spikes = trends_df.filter((col("is_spike") == True) & (col("reaction_name") != "ALL")).count()

    print(f"Total monthly trend time-series records: {total_records:,}")
    print(f"Total anomalous spikes detected (Z >= 3.0, N >= 3): {spike_records:,}")
    print(f"  - Overall drug volume spikes: {drug_spikes:,}")
    print(f"  - Reaction-specific signal spikes: {signal_spikes:,}")

    # 4. Save to Parquet
    print(f"Saving trends dataset to {output_path}...")
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path, ignore_errors=True)
        else:
            output_path.unlink(missing_ok=True)

    try:
        trends_df.coalesce(2).write.mode("overwrite").parquet(str(output_path.as_posix()))
    except Exception as write_err:
        print(f"Spark direct write note: {write_err}. Saving via PyArrow...")
        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path, ignore_errors=True)
            else:
                output_path.unlink(missing_ok=True)
        pdf = trends_df.toPandas()
        pdf.to_parquet(output_path, engine="pyarrow", index=False)

    elapsed = time.time() - start_time
    print(f"Trends analysis completed in {elapsed:.1f}s.")
    print(f"Parquet table saved at: {output_path}")

    # Preview top spikes
    print("\nTop 10 Adverse Event Volume Spikes (highest Z-score):")
    (
        trends_df.filter(col("is_spike") == True)
        .select("drug_name", "reaction_name", "year_month", "report_count", "mean_count", "z_score")
        .orderBy(col("z_score").desc())
        .show(10, truncate=False)
    )

    spark.stop()


if __name__ == "__main__":
    compute_safety_trends()

