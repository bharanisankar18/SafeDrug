"""
SafeDose - Step 4: Safety Signal Detection Job
Calculates pharmacovigilance statistical disproportionality metrics:
- Proportional Reporting Ratio (PRR)
- Reporting Odds Ratio (ROR) with 95% Confidence Interval
- Chi-Square statistic (with Yates' continuity correction)
- Evans Signal Criteria Flag: (count >= 3) & (PRR >= 2.0) & (chi_square >= 4.0)

Input: data/processed/clean_reports.parquet
Output: data/processed/signals.parquet
"""

import sys
import time
import shutil
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spark_init import PROCESSED_DATA_DIR, get_spark_session
from pyspark.sql.functions import (
    col, count, countDistinct, lit, when, exp, sqrt, log,
    greatest, round as spark_round
)


def compute_safety_signals():
    start_time = time.time()
    print("=== SafeDose Step 4: Safety Signal Detection Job ===")

    # 1. Initialize Spark session
    spark = get_spark_session(app_name="SafeDose-SignalDetection", memory="6g")

    clean_parquet_path = PROCESSED_DATA_DIR / "clean_reports.parquet"
    if not clean_parquet_path.exists():
        raise FileNotFoundError(f"Clean reports table not found at {clean_parquet_path}. Run Step 3 first.")

    print(f"Reading clean reports from {clean_parquet_path}...")
    df = spark.read.parquet(str(clean_parquet_path.as_posix()))

    # Total unique reports in the entire dataset (N)
    total_reports = df.select("safetyreportid").distinct().count()
    print(f"Total distinct reports in dataset (N): {total_reports:,}")

    # 2. Count reports per drug: N_drug = a + b
    print("Calculating drug-level report counts...")
    drug_counts = (
        df.groupBy("drug_name")
        .agg(countDistinct("safetyreportid").alias("n_drug"))
    )

    # 3. Count reports per reaction: N_reaction = a + c
    print("Calculating reaction-level report counts...")
    reaction_counts = (
        df.groupBy("reaction_name")
        .agg(countDistinct("safetyreportid").alias("n_reaction"))
    )

    # 4. Count reports per (drug, reaction) pair: a
    print("Calculating drug-reaction pair report counts (a)...")
    pair_counts = (
        df.groupBy("drug_name", "reaction_name")
        .agg(countDistinct("safetyreportid").alias("a"))
    )

    # 5. Join to construct 2x2 contingency table for every pair
    print("Building 2x2 contingency tables...")
    contingency = (
        pair_counts
        .join(drug_counts, on="drug_name", how="inner")
        .join(reaction_counts, on="reaction_name", how="inner")
        .withColumn("total_reports", lit(total_reports))
        # b = reports with drug_name but without reaction_name
        .withColumn("b", col("n_drug") - col("a"))
        # c = reports with reaction_name but other drugs
        .withColumn("c", col("n_reaction") - col("a"))
        # d = reports with neither drug_name nor reaction_name
        .withColumn("d", lit(total_reports) - col("a") - col("b") - col("c"))
    )

    # 6. Calculate PRR, ROR, and Chi-Square with continuity correction
    print("Calculating PRR, ROR (95% CI), and Yates Chi-Square statistics...")
    
    # Proportions
    prop_drug = col("a") / (col("a") + col("b"))
    prop_other = col("c") / (col("c") + col("d"))

    # PRR = [a / (a + b)] / [c / (c + d)]
    prr_expr = when(
        (col("c") > 0) & (prop_other > 0),
        prop_drug / prop_other
    ).otherwise(lit(None))

    # ROR = (a * d) / (b * c)
    ror_expr = when(
        (col("b") > 0) & (col("c") > 0),
        (col("a") * col("d")) / (col("b") * col("c"))
    ).otherwise(lit(None))

    # SE(ln(ROR)) = sqrt(1/a + 1/b + 1/c + 1/d)
    se_ln_ror = sqrt(
        (lit(1.0) / col("a")) +
        (lit(1.0) / greatest(col("b"), lit(1.0))) +
        (lit(1.0) / greatest(col("c"), lit(1.0))) +
        (lit(1.0) / greatest(col("d"), lit(1.0)))
    )

    # ROR 95% Confidence Intervals
    ror_lower_expr = when(
        ror_expr.isNotNull(),
        exp(log(ror_expr) - (lit(1.96) * se_ln_ror))
    ).otherwise(lit(None))

    ror_upper_expr = when(
        ror_expr.isNotNull(),
        exp(log(ror_expr) + (lit(1.96) * se_ln_ror))
    ).otherwise(lit(None))

    # Yates Chi-Square: N * (|a*d - b*c| - N/2)^2 / [(a+b)(c+d)(a+c)(b+d)]
    ad_minus_bc = col("a") * col("d") - col("b") * col("c")
    abs_diff = when(ad_minus_bc < 0, -ad_minus_bc).otherwise(ad_minus_bc)
    yates_diff = greatest(lit(0.0), abs_diff - (lit(total_reports) / lit(2.0)))
    
    denominator = (
        (col("a") + col("b")).cast("double") *
        (col("c") + col("d")).cast("double") *
        (col("a") + col("c")).cast("double") *
        (col("b") + col("d")).cast("double")
    )

    chi_square_expr = when(
        denominator > 0,
        (lit(total_reports).cast("double") * (yates_diff * yates_diff)) / denominator
    ).otherwise(lit(0.0))

    signals_df = (
        contingency
        .withColumn("prr", spark_round(prr_expr, 3))
        .withColumn("ror", spark_round(ror_expr, 3))
        .withColumn("ror_lower_ci", spark_round(ror_lower_expr, 3))
        .withColumn("ror_upper_ci", spark_round(ror_upper_expr, 3))
        .withColumn("chi_square", spark_round(chi_square_expr, 3))
        # Flag if PRR >= 2.0, chi_square >= 4.0, and a >= 3
        .withColumn(
            "is_signal",
            (col("a") >= 3) &
            (col("prr").isNotNull()) & (col("prr") >= 2.0) &
            (col("chi_square").isNotNull()) & (col("chi_square") >= 4.0)
        )
        .select(
            col("drug_name"),
            col("reaction_name"),
            col("a").alias("report_count"),
            col("n_drug").alias("total_drug_reports"),
            col("n_reaction").alias("total_reaction_reports"),
            col("b"),
            col("c"),
            col("d"),
            col("prr"),
            col("ror"),
            col("ror_lower_ci"),
            col("ror_upper_ci"),
            col("chi_square"),
            col("is_signal")
        )
    )

    # 7. Metrics Summary
    total_pairs = signals_df.count()
    flagged_signals = signals_df.filter(col("is_signal") == True).count()
    print(f"Total drug-reaction pairs evaluated: {total_pairs:,}")
    print(f"Statistical safety signals detected (Evans criteria): {flagged_signals:,}")

    # 8. Save as Parquet
    output_path = PROCESSED_DATA_DIR / "signals.parquet"
    print(f"Saving signals dataset to {output_path}...")
    
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path, ignore_errors=True)
        else:
            output_path.unlink(missing_ok=True)

    try:
        signals_df.coalesce(4).write.mode("overwrite").parquet(str(output_path.as_posix()))
    except Exception as write_err:
        print(f"Direct Spark write note: {write_err}")
        print("Writing via optimized PyArrow engine...")
        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path, ignore_errors=True)
            else:
                output_path.unlink(missing_ok=True)
        pdf = signals_df.toPandas()
        pdf.to_parquet(output_path, engine="pyarrow", index=False)

    elapsed = time.time() - start_time
    print(f"Signal detection completed in {elapsed:.1f}s.")
    print(f"Parquet table saved at: {output_path}")

    # Show top detected signals
    print("\nTop 10 Detected Safety Signals (by report count):")
    (
        signals_df.filter(col("is_signal") == True)
        .orderBy(col("report_count").desc(), col("chi_square").desc())
        .select("drug_name", "reaction_name", "report_count", "prr", "ror", "chi_square")
        .show(10, truncate=False)
    )

    spark.stop()
    return output_path, total_pairs, flagged_signals


if __name__ == "__main__":
    compute_safety_signals()

