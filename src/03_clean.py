"""
SafeDose - Step 3: Data Cleaning & Preprocessing Job
Processes raw OpenFDA adverse event reports with Apache Spark:
- Deduplicates reports by safetyreportid
- Keeps suspect drugs (characterization == '1')
- Standardizes drug and reaction names (MedDRA terms)
- Normalizes patient demographics (age, sex, seriousness, date)
- Saves the clean tabular dataset to data/processed/clean_reports.parquet
"""

import sys
import time
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spark_init import RAW_DATA_DIR, PROCESSED_DATA_DIR, get_spark_session
from pyspark.sql.functions import col, explode, lower, trim, when, size, coalesce, to_date


def clean_adverse_events():
    start_time = time.time()
    print("=== SafeDose Step 3: Data Cleaning Job ===")
    
    # 1. Initialize Spark session
    spark = get_spark_session(app_name="SafeDose-CleanReports", memory="6g")
    
    # 2. Locate raw JSON files
    json_files = sorted(list(RAW_DATA_DIR.glob("*.json")))
    if not json_files:
        raise FileNotFoundError(f"No raw JSON files found in {RAW_DATA_DIR}")
        
    print(f"Reading {len(json_files)} raw JSON files...")
    file_paths = [f.as_posix() for f in json_files]
    
    # Load multi-line JSON files
    raw_df = (
        spark.read
        .option("multiline", "true")
        .json(file_paths)
    )
    
    # 3. OpenFDA data wraps reports inside a 'results' array
    if "results" in raw_df.columns:
        print("Unpacking results array from OpenFDA payload...")
        df = raw_df.select(explode(col("results")).alias("report")).select("report.*")
    else:
        df = raw_df

    raw_report_count = df.count()
    print(f"Total raw reports loaded: {raw_report_count:,}")

    # 4. Remove duplicate reports by safetyreportid
    df_dedup = df.dropDuplicates(["safetyreportid"])
    unique_report_count = df_dedup.count()
    print(f"Unique reports after deduplication: {unique_report_count:,} (removed {raw_report_count - unique_report_count:,} duplicates)")

    # 5. Explode drugs and keep Primary Suspect drugs (drugcharacterization == '1')
    df_drugs = df_dedup.withColumn("drug", explode(col("patient.drug")))
    df_suspect = df_drugs.filter(col("drug.drugcharacterization") == "1")

    # 6. Standardize drug names:
    # Prefer standardized generic_name from OpenFDA if present; fallback to medicinalproduct
    has_generic = (col("drug.openfda.generic_name").isNotNull()) & (size(col("drug.openfda.generic_name")) > 0)
    std_drug_name = when(
        has_generic,
        lower(trim(col("drug.openfda.generic_name")[0]))
    ).otherwise(lower(trim(col("drug.medicinalproduct"))))

    df_drugs_std = (
        df_suspect
        .withColumn("drug_name", std_drug_name)
        .withColumn("medicinalproduct", lower(trim(col("drug.medicinalproduct"))))
    )

    # 7. Explode reactions and standardize reaction MedDRA Preferred Terms
    df_reactions = df_drugs_std.withColumn("reaction", explode(col("patient.reaction")))
    df_reactions_std = df_reactions.withColumn(
        "reaction_name",
        lower(trim(col("reaction.reactionmeddrapt")))
    )

    # 8. Select and cast required fields
    clean_df = (
        df_reactions_std
        .select(
            col("safetyreportid").cast("string").alias("safetyreportid"),
            col("receivedate").cast("string").alias("receivedate"),
            col("serious").cast("string").alias("seriousness"),
            col("patient.patientsex").cast("string").alias("sex"),
            col("patient.patientonsetage").cast("double").alias("age"),
            col("drug_name"),
            col("medicinalproduct"),
            col("reaction_name")
        )
        .dropna(subset=["safetyreportid", "drug_name", "reaction_name"])
        .filter((col("drug_name") != "") & (col("reaction_name") != ""))
        # Deduplicate identical report-drug-reaction rows
        .dropDuplicates(["safetyreportid", "drug_name", "reaction_name"])
    )

    # 9. Count clean records
    clean_count = clean_df.count()
    print(f"Clean drug-reaction records: {clean_count:,}")

    # 10. Save as Parquet
    output_path = PROCESSED_DATA_DIR / "clean_reports.parquet"
    print(f"Writing clean dataset ({clean_count:,} rows) to {output_path}...")
    
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path, ignore_errors=True)
        else:
            output_path.unlink(missing_ok=True)

    try:
        clean_df.coalesce(4).write.mode("overwrite").parquet(str(output_path.as_posix()))
    except Exception as write_err:
        print(f"Direct Spark parquet write note: {write_err}")
        print("Writing via optimized PyArrow engine...")
        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path, ignore_errors=True)
            else:
                output_path.unlink(missing_ok=True)
        pdf = clean_df.toPandas()
        pdf.to_parquet(output_path, engine="pyarrow", index=False)

    elapsed = time.time() - start_time
    print(f"Cleaning completed successfully in {elapsed:.1f}s.")
    print(f"Parquet table saved at: {output_path}")

    # Print summary sample
    print("\nSample clean records:")
    clean_df.select("safetyreportid", "receivedate", "sex", "age", "drug_name", "reaction_name").show(5, truncate=False)

    spark.stop()
    return output_path, clean_count, unique_report_count


if __name__ == "__main__":
    clean_adverse_events()
