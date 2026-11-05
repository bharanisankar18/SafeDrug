"""
SafeDose - Step 2: Download & Ingestion Script
Downloads OpenFDA FAERS quarterly adverse drug event files,
unzips them into data/raw, and validates raw record counts.
"""

import os
import sys
import shutil
import zipfile
import requests
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spark_init import RAW_DATA_DIR, get_spark_session


OPENFDA_API_URL = "https://api.fda.gov/download.json"


def normalize_raw_directory(raw_dir: Path):
    """
    If files were unzipped into subdirectories with duplicate names
    (e.g., data/raw/drug-event-0001.json/drug-event-0001.json),
    moves them to data/raw/ directly and cleans up the empty folder.
    """
    for item in list(raw_dir.iterdir()):
        if item.is_dir() and item.name.endswith(".json"):
            nested_file = item / item.name
            target_path = raw_dir / item.name
            temp_name = raw_dir / f"temp_{item.name}"
            
            if nested_file.exists():
                print(f"Normalizing nested file: {nested_file.name} -> {raw_dir}")
                shutil.move(str(nested_file), str(temp_name))
                shutil.rmtree(str(item), ignore_errors=True)
                shutil.move(str(temp_name), str(target_path))


def fetch_available_partitions(limit: int = 4):
    """
    Fetches download URLs for drug adverse event reports from OpenFDA API.
    """
    try:
        response = requests.get(OPENFDA_API_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
        drug_events = data.get("results", {}).get("drug", {}).get("event", {})
        partitions = drug_events.get("partitions", [])
        return partitions[:limit]
    except Exception as e:
        print(f"Warning: Could not fetch OpenFDA index ({e}).")
        return []


def download_and_extract_file(url: str, output_dir: Path):
    """
    Downloads a zip file from OpenFDA and extracts its JSON contents into output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    zip_path = output_dir / filename
    
    print(f"Downloading {filename} from {url}...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
        
    # Clean up zip file
    zip_path.unlink()
    print(f"Extracted and removed {filename}")


def inspect_raw_files(raw_dir: Path):
    """
    Finds all raw JSON files and counts reports using PySpark with multiline=true.
    """
    normalize_raw_directory(raw_dir)
    json_files = sorted(list(raw_dir.glob("*.json")))
    
    if not json_files:
        print(f"No JSON files found directly in {raw_dir}. Checking subdirectories...")
        json_files = sorted(list(raw_dir.glob("**/*.json")))
        
    if not json_files:
        print(f"No raw JSON files found in {raw_dir}.")
        return [], 0

    print("\n" + "=" * 65)
    print(f"Found {len(json_files)} raw JSON partition file(s) in {raw_dir}:")
    total_size_mb = 0.0
    file_info = []
    for idx, f in enumerate(json_files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size_mb += size_mb
        file_info.append({"name": f.name, "path": str(f), "size_mb": round(size_mb, 2)})
        print(f"  {idx}. {f.name} ({size_mb:.2f} MB)")
    print(f"Total raw data size: {total_size_mb:.2f} MB (~{total_size_mb/1024:.2f} GB)")
    print("=" * 65)
    
    print("\nCounting raw report records with Apache Spark (multiline=True)...")
    spark = get_spark_session("Step2-RawDataInspect", memory="6g")
    
    # Pass explicit POSIX path strings to prevent Windows hadoop globbing errors
    file_paths = [f.as_posix() for f in json_files]
    
    from pyspark.sql.functions import col, explode
    
    raw_df = (
        spark.read
        .option("multiline", "true")
        .json(file_paths)
    )
    
    if "results" in raw_df.columns:
        reports_df = raw_df.select(explode(col("results")).alias("report"))
        total_reports = reports_df.count()
    else:
        total_reports = raw_df.count()
        
    print(f"\n>>> Total raw adverse event reports across all files: {total_reports:,} <<<")
    spark.stop()
    
    return file_info, total_reports


if __name__ == "__main__":
    print("=== SafeDose Step 2: Download & Raw Data Verification ===")
    files, total_count = inspect_raw_files(RAW_DATA_DIR)
    print(f"\nStep 2 Status: Successfully verified {len(files)} raw files with {total_count:,} adverse event reports.")
