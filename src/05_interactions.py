"""
SafeDose - Step 5: Drug Interactions Analysis (FP-Growth Association Mining)
Uses Apache Spark MLlib FP-Growth to discover co-prescribed / co-occurring drug pairs
and flags drug pairs linked to unusually high reaction rates or serious outcomes.

Input: data/processed/clean_reports.parquet
Output: data/processed/interactions.parquet
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
    col, collect_set, countDistinct, count, size, max as spark_max,
    avg as spark_avg, when, lit, array_contains, round as spark_round,
    first, desc, row_number, broadcast
)
from pyspark.sql.window import Window
from pyspark.ml.fpm import FPGrowth


def mine_drug_interactions():
    start_time = time.time()
    print("=== SafeDose Step 5: Drug Interactions Mining Job ===")

    spark = get_spark_session(app_name="SafeDose-InteractionsMining", memory="6g")

    clean_parquet_path = PROCESSED_DATA_DIR / "clean_reports.parquet"
    if not clean_parquet_path.exists():
        raise FileNotFoundError(f"Clean reports table not found at {clean_parquet_path}")

    print(f"Loading clean reports from {clean_parquet_path}...")
    df = spark.read.parquet(str(clean_parquet_path.as_posix()))

    # 1. Report-level aggregations
    print("Aggregating multi-drug baskets per report...")
    report_summary = (
        df.groupBy("safetyreportid")
        .agg(
            collect_set("drug_name").alias("drugs"),
            countDistinct("reaction_name").alias("num_reactions"),
            spark_max(when(col("seriousness") == "1", 1).otherwise(0)).alias("is_serious")
        )
    )

    total_multi_reports = report_summary.filter(size("drugs") >= 2).count()
    global_avg_rx = report_summary.select(spark_avg("num_reactions")).collect()[0][0]
    print(f"Total multi-drug patient reports: {total_multi_reports:,}")
    print(f"Global baseline reaction count per patient: {global_avg_rx:.2f}")

    # 2. Single drug baselines (average reactions when taking drug)
    print("Calculating individual drug reaction baselines...")
    drug_baselines = (
        df.groupBy("drug_name")
        .agg(
            countDistinct("safetyreportid").alias("single_drug_reports"),
            count("reaction_name").alias("total_reactions")
        )
        .withColumn(
            "drug_avg_reactions",
            col("total_reactions") / col("single_drug_reports")
        )
    )

    # 3. FP-Growth frequent itemset mining on drug combinations
    print("Running Spark MLlib FP-Growth (minSupport=0.003, minConfidence=0.1)...")
    baskets = report_summary.filter(size("drugs") >= 2)
    fp = FPGrowth(itemsCol="drugs", minSupport=0.003, minConfidence=0.1)
    model = fp.fit(baskets)

    # Filter for 2-item sets (drug pairs)
    pairs_df = (
        model.freqItemsets
        .filter(size(col("items")) == 2)
        .withColumn("drug_1", col("items")[0])
        .withColumn("drug_2", col("items")[1])
        .withColumn("pair_report_count", col("freq"))
        .withColumn("support", spark_round(col("freq") / lit(total_multi_reports), 4))
    )

    num_pairs = pairs_df.count()
    print(f"Frequent co-occurring drug pairs identified: {num_pairs:,}")

    # 4. Analyze adverse reaction rates for each drug pair
    # Join reports to drug pairs to calculate actual reaction rates when taking both
    print("Analyzing adverse reaction severity and excess rates per drug pair...")
    
    # Distinct drug assignments per report (avoids multiplying by reaction count)
    report_drugs = df.select("safetyreportid", "drug_name").distinct()
    # Distinct reactions & seriousness per report
    report_reactions = df.select("safetyreportid", "reaction_name", "seriousness").distinct()

    # Form unique drug pairs present in each multi-drug report
    pair_reports = (
        report_drugs.alias("a")
        .join(report_drugs.alias("b"), on="safetyreportid")
        .filter(col("a.drug_name") < col("b.drug_name"))
        .select(
            col("safetyreportid"),
            col("a.drug_name").alias("d1"),
            col("b.drug_name").alias("d2")
        )
    )

    # Filter only for the frequent pairs discovered by FP-Growth (broadcast 561 rows)
    ordered_pairs = pairs_df.withColumn(
        "d1", when(col("drug_1") < col("drug_2"), col("drug_1")).otherwise(col("drug_2"))
    ).withColumn(
        "d2", when(col("drug_1") < col("drug_2"), col("drug_2")).otherwise(col("drug_1"))
    ).select("d1", "d2", "pair_report_count", "support")

    pair_matched = (
        broadcast(ordered_pairs)
        .join(pair_reports, on=["d1", "d2"], how="inner")
        .join(report_reactions, on="safetyreportid")
    ).cache()

    # Pair metrics: total reactions, distinct reports, serious count
    pair_stats = (
        pair_matched
        .groupBy("d1", "d2", "pair_report_count", "support")
        .agg(
            count("reaction_name").alias("total_pair_reactions"),
            countDistinct("safetyreportid").alias("matched_reports"),
            spark_max(when(col("seriousness") == "1", 1).otherwise(0)).alias("has_serious"),
            count(when(col("seriousness") == "1", 1)).alias("serious_record_count")
        )
        .withColumn(
            "avg_reaction_count",
            spark_round(col("total_pair_reactions") / col("pair_report_count"), 2)
        )
        .withColumn(
            "serious_rate",
            spark_round(col("serious_record_count") / col("total_pair_reactions"), 3)
        )
    )

    # Find top reaction per pair
    rx_window = Window.partitionBy("d1", "d2").orderBy(desc("reaction_count"))
    top_rx_df = (
        pair_matched
        .groupBy("d1", "d2", "reaction_name")
        .agg(count("safetyreportid").alias("reaction_count"))
        .withColumn("rank", row_number().over(rx_window))
        .filter(col("rank") == 1)
        .select(
            "d1", "d2",
            col("reaction_name").alias("top_reaction"),
            col("reaction_count").alias("top_reaction_count")
        )
    )

    # 5. Join baseline metrics and calculate Reaction Rate Ratio
    baseline_d1 = drug_baselines.select(
        col("drug_name").alias("d1"),
        col("drug_avg_reactions").alias("d1_avg_rx")
    )
    baseline_d2 = drug_baselines.select(
        col("drug_name").alias("d2"),
        col("drug_avg_reactions").alias("d2_avg_rx")
    )

    final_interactions = (
        pair_stats
        .join(top_rx_df, on=["d1", "d2"], how="left")
        .join(baseline_d1, on="d1", how="left")
        .join(baseline_d2, on="d2", how="left")
        .withColumn(
            "baseline_avg_reactions",
            spark_round((col("d1_avg_rx") + col("d2_avg_rx")) / lit(2.0), 2)
        )
        .withColumn(
            "reaction_rate_ratio",
            spark_round(col("avg_reaction_count") / col("baseline_avg_reactions"), 3)
        )
        # Flag unusually high reaction rates or disproportionate serious events
        .withColumn(
            "is_high_risk_interaction",
            (col("reaction_rate_ratio") >= 1.20) |
            (col("serious_rate") >= 0.70)
        )
        .select(
            col("d1").alias("drug_1"),
            col("d2").alias("drug_2"),
            col("pair_report_count"),
            col("support"),
            col("avg_reaction_count"),
            col("baseline_avg_reactions"),
            col("reaction_rate_ratio"),
            col("serious_rate"),
            col("top_reaction"),
            col("top_reaction_count"),
            col("is_high_risk_interaction")
        )
    )

    total_interactions = final_interactions.count()
    high_risk_count = final_interactions.filter(col("is_high_risk_interaction") == True).count()
    print(f"Total analyzed drug pairs: {total_interactions:,}")
    print(f"High-risk interaction pairs flagged: {high_risk_count:,}")

    # 6. Save as Parquet
    output_path = PROCESSED_DATA_DIR / "interactions.parquet"
    print(f"Writing interactions table to {output_path}...")

    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path, ignore_errors=True)
        else:
            output_path.unlink(missing_ok=True)

    try:
        final_interactions.coalesce(2).write.mode("overwrite").parquet(str(output_path.as_posix()))
    except Exception as write_err:
        print(f"Direct Spark write note: {write_err}. Saving via PyArrow...")
        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path, ignore_errors=True)
            else:
                output_path.unlink(missing_ok=True)
        pdf = final_interactions.toPandas()
        pdf.to_parquet(output_path, engine="pyarrow", index=False)

    elapsed = time.time() - start_time
    print(f"Interactions mining completed in {elapsed:.1f}s.")
    print(f"Parquet table saved at: {output_path}")

    print("\nTop 10 High-Risk Drug Interactions (by reaction rate ratio):")
    (
        final_interactions.filter(col("is_high_risk_interaction") == True)
        .orderBy(desc("reaction_rate_ratio"), desc("pair_report_count"))
        .select(
            "drug_1", "drug_2", "pair_report_count",
            "avg_reaction_count", "reaction_rate_ratio", "serious_rate", "top_reaction"
        )
        .show(10, truncate=False)
    )

    spark.stop()
    return output_path, total_interactions, high_risk_count


if __name__ == "__main__":
    mine_drug_interactions()

